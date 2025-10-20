import gc 
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import cuda
import os
import argparse
import dask.array as da
from daskms import xds_from_ms, xds_from_table, xds_to_table
import multiprocessing
from tqdm import tqdm
import xarray as xr
import dask
import psutil
import concurrent.futures

def apply_flags_to_ms(ms_file, bl, field_id, new_flags):
    """Apply flags to the MS file for a specific baseline"""
    # Create baseline-specific query
    bl_taql = f"FIELD_ID={field_id} AND ANTENNA1={bl[0]} AND ANTENNA2={bl[1]}"
    
    # Read original data
    orig_ds_list = xds_from_ms(
        ms_file,
        columns=("FLAG",),
        taql_where=bl_taql
    )
    
    if not orig_ds_list or orig_ds_list[0].sizes['row'] == 0:
        print(f"No data found for baseline {bl} when writing flags. Skipping.")
        return False
    
    try:
        # Get original dataset and flags
        orig_ds = orig_ds_list[0]
        orig_flags = orig_ds.FLAG.data.compute()
        
        # Ensure new_flags matches the shape of orig_flags
        if orig_flags.shape != new_flags.shape:
            print(f"Shape mismatch: Original {orig_flags.shape}, New {new_flags.shape}")
            
            # Create properly shaped combined flags
            combined_flags = orig_flags.copy()
            
            # Handle the most common case: 2D flags need to be broadcast to 3D
            if len(new_flags.shape) == 2 and len(orig_flags.shape) == 3:
                # Apply to all correlations
                for corr_idx in range(orig_flags.shape[2]):
                    # Only update rows that exist in both arrays
                    min_rows = min(orig_flags.shape[0], new_flags.shape[0])
                    min_chans = min(orig_flags.shape[1], new_flags.shape[1])
                    
                    # OR the flags together
                    combined_flags[:min_rows, :min_chans, corr_idx] = np.logical_or(
                        orig_flags[:min_rows, :min_chans, corr_idx],
                        new_flags[:min_rows, :min_chans]
                    )
            else:
                # For other dimension mismatches, just use common dimensions
                common_dims = tuple(min(d1, d2) for d1, d2 in zip(orig_flags.shape, new_flags.shape))
                slices = tuple(slice(0, d) for d in common_dims)
                
                combined_flags[slices] = np.logical_or(
                    orig_flags[slices],
                    new_flags[slices]
                )
        else:
            # Shapes match, simple OR operation
            combined_flags = np.logical_or(orig_flags, new_flags)
        
        # Convert to dask array with the SAME chunking as original
        # This is crucial for proper writing
        new_flags_dask = da.from_array(combined_flags, chunks=orig_ds.FLAG.data.chunks)
        
        # Create updated dataset
        updated_ds = orig_ds.assign(FLAG=(orig_ds.FLAG.dims, new_flags_dask))
        
        # Write back
        print(f"Writing flags for baseline {bl}...")
        write_back = xds_to_table([updated_ds], ms_file, ["FLAG"])
        dask.compute(write_back)
        
        return True
    except Exception as e:
        print(f"Error writing flags for baseline {bl}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def print_gpu_info():
    """Print basic information about available GPUs"""
    print("=== GPU Information ===")
    try:
        devices = cuda.list_devices()
        print(f"Number of available GPUs: {len(devices)}")
        
        device = cuda.get_current_device()
        print(f"GPU 0: {device.name}")
        
        # Get compute capability
        try:
            cc = device.compute_capability
            print(f"  - Compute capability: {cc[0]}.{cc[1]}")
        except:
            print("  - Compute capability info not available")
    except Exception as e:
        print(f"Error getting GPU info: {str(e)}")
    
    print("======================")

@cuda.jit
def sum_threshold_kernel_time_channel(amp, flags, thresholds, M):
    """Improved CUDA kernel for SumThreshold in time direction with better flagging logic"""
    i, j = cuda.grid(2)
    
    if i < amp.shape[0] and j < amp.shape[1] - M + 1:
        # Calculate average threshold for this group
        avg_threshold = 0.0
        for k in range(M):
            avg_threshold += thresholds[j+k]
        avg_threshold /= M
        
        # Count unflagged points and calculate their sum
        group_sum = 0.0
        count = 0
        
        for k in range(M):
            if not flags[i, j+k]:
                group_sum += amp[i, j+k]
                count += 1
        
        # Only proceed if we have enough unflagged points (at least 30%)
        min_unflagged = max(1, int(M * 0.3))
        
        if count >= min_unflagged:
            # Check if the average exceeds the threshold
            if (group_sum / count) > avg_threshold:
                # Flag samples that exceed the threshold
                for k in range(M):
                    if not flags[i, j+k]:
                        # Flag more conservatively for low amplitudes
                        if amp[i, j+k] < avg_threshold * 0.5:
                            # For very low amplitudes, require higher confidence
                            if amp[i, j+k] > thresholds[j+k] * 1.2:
                                flags[i, j+k] = True
                        else:
                            # For normal/high amplitudes, use standard threshold
                            if amp[i, j+k] > thresholds[j+k]:
                                flags[i, j+k] = True

@cuda.jit
def sum_threshold_kernel_freq_channel(amp, flags, thresholds, M):
    """Improved CUDA kernel for SumThreshold in frequency direction with better flagging logic"""
    i, j = cuda.grid(2)
    
    if i < amp.shape[0] - M + 1 and j < amp.shape[1]:
        # Get threshold for this channel
        threshold = thresholds[j]
        
        # Count unflagged points and calculate their sum
        group_sum = 0.0
        count = 0
        
        for k in range(M):
            if not flags[i+k, j]:
                group_sum += amp[i+k, j]
                count += 1
        
        # Only proceed if we have enough unflagged points (at least 30%)
        min_unflagged = max(1, int(M * 0.3))
        
        if count >= min_unflagged:
            # Check if the average exceeds the threshold
            if (group_sum / count) > threshold:
                # Flag samples that exceed the threshold
                for k in range(M):
                    if not flags[i+k, j]:
                        # Flag more conservatively for low amplitudes
                        if amp[i+k, j] < threshold * 0.5:
                            # For very low amplitudes, require higher confidence
                            if amp[i+k, j] > threshold * 1.2:
                                flags[i+k, j] = True
                        else:
                            # For normal/high amplitudes, use standard threshold
                            if amp[i+k, j] > threshold:
                                flags[i+k, j] = True

def calculate_robust_thresholds(amp, flags, sigma_factor=6.0):
    """Calculate robust thresholds using median and MAD instead of mean and std"""
    channel_thresholds = np.zeros(amp.shape[1], dtype=np.float32)
    
    for j in range(amp.shape[1]):
        # Get unflagged data for this channel
        channel_data = amp[:, j]
        mask = flags[:, j]
        unflagged_data = channel_data[~mask]
        
        if len(unflagged_data) > 0:
            # Calculate median
            median = np.median(unflagged_data)
            # Use sigma_factor as a multiplier for the median
            channel_thresholds[j] = median * sigma_factor
        else:
            # If all data is flagged in this channel, use surrounding channels or defaults
            nearby_channels = []
            for offset in range(1, min(5, amp.shape[1])):
                if j-offset >= 0:
                    nearby_j = j-offset
                    nearby_data = amp[:, nearby_j]
                    nearby_mask = flags[:, nearby_j]
                    nearby_unflagged = nearby_data[~nearby_mask]
                    if len(nearby_unflagged) > 0:
                        nearby_channels.append(nearby_unflagged)
                
                if j+offset < amp.shape[1]:
                    nearby_j = j+offset
                    nearby_data = amp[:, nearby_j]
                    nearby_mask = flags[:, nearby_j]
                    nearby_unflagged = nearby_data[~nearby_mask]
                    if len(nearby_unflagged) > 0:
                        nearby_channels.append(nearby_unflagged)
            
            if nearby_channels:
                # Use statistics from nearby channels
                all_nearby_data = np.concatenate(nearby_channels)
                median = np.median(all_nearby_data)
                channel_thresholds[j] = median * sigma_factor
            else:
                # Fallback to reasonable default
                channel_thresholds[j] = np.median(amp) * sigma_factor
    
    return channel_thresholds




@cuda.jit
def bandpass_normalize_kernel(amp, flags, rfi_threshold):
    """
    CUDA kernel to normalize bandpass and flag RFI channels.
    
    Args:
        amp: Amplitude array (2D: time × frequency) - modified in-place
        flags: Boolean array of existing flags (same shape as amp) - modified in-place
        rfi_threshold: Threshold for RFI detection
    """
    # Get channel index
    channel_idx = cuda.grid(1)
    
    # Process only valid channels
    if channel_idx < amp.shape[1]:
        # Calculate median for this channel
        values = cuda.local.array(1000, dtype=np.float32)  # Adjust size as needed
        count = 0
        
        # Collect unflagged values
        for t in range(amp.shape[0]):
            if not flags[t, channel_idx] and count < 1000:
                values[count] = amp[t, channel_idx]
                count += 1
        
        # Calculate bandpass (median)
        bandpass_value = 0.0
        if count > 0:
            # Simple sort
            for i in range(count):
                for j in range(i+1, count):
                    if values[i] > values[j]:
                        values[i], values[j] = values[j], values[i]
            
            # Get median
            if count % 2 == 0:
                bandpass_value = (values[count//2-1] + values[count//2]) / 2
            else:
                bandpass_value = values[count//2]
        
        # Check if channel is RFI-affected by comparing with neighbors
        is_rfi = False
        if bandpass_value > 0:
            # Calculate neighbor statistics (simple approach)
            neighbor_sum = 0.0
            neighbor_count = 0
            
            # Check 3 channels to the left and right
            for offset in range(-3, 4):
                if offset != 0:  # Skip current channel
                    neighbor_idx = channel_idx + offset
                    if 0 <= neighbor_idx < amp.shape[1]:
                        # Get bandpass for this neighbor by sampling a few time points
                        neighbor_vals = cuda.local.array(10, dtype=np.float32)
                        n_count = 0
                        
                        # Sample a few time points
                        for t_step in range(0, amp.shape[0], amp.shape[0]//10 + 1):
                            if t_step < amp.shape[0] and not flags[t_step, neighbor_idx] and n_count < 10:
                                neighbor_vals[n_count] = amp[t_step, neighbor_idx]
                                n_count += 1
                        
                        # Get median of samples
                        if n_count > 0:
                            # Sort
                            for i in range(n_count):
                                for j in range(i+1, n_count):
                                    if neighbor_vals[i] > neighbor_vals[j]:
                                        neighbor_vals[i], neighbor_vals[j] = neighbor_vals[j], neighbor_vals[i]
                            
                            # Add to statistics
                            if n_count % 2 == 0:
                                neighbor_sum += (neighbor_vals[n_count//2-1] + neighbor_vals[n_count//2]) / 2
                            else:
                                neighbor_sum += neighbor_vals[n_count//2]
                            neighbor_count += 1
            
            # Check if this channel is an outlier
            if neighbor_count > 0:
                neighbor_avg = neighbor_sum / neighbor_count
                if bandpass_value > neighbor_avg * rfi_threshold:
                    is_rfi = True
            
            # Normalize or flag
            for t in range(amp.shape[0]):
                if is_rfi:
                    # Flag RFI channel
                    flags[t, channel_idx] = True
                elif not flags[t, channel_idx]:
                    # Normalize by bandpass
                    amp[t, channel_idx] = amp[t, channel_idx] / bandpass_value

def sumthreshold_gpu(amp, existing_flags, baseline_info, combinations=None, sigma_factor=6.0, rho=1.5, 
                    diagnostic_plots=False, stream=None, precalculated_thresholds=None):
    """
    SumThreshold implementation with integrated bandpass normalization
    
    Args:
        amp: Amplitude array (2D: time × frequency)
        existing_flags: Boolean array of existing flags (same shape as amp)
        baseline_info: String description of the baseline for logging
        combinations: List of window sizes to use for SumThreshold
        sigma_factor: Multiplier for standard deviation in threshold calculation (not used with precalculated_thresholds)
        rho: Factor to reduce threshold for larger window sizes
        diagnostic_plots: Whether to generate diagnostic plots
        stream: CUDA stream to use for asynchronous execution
        precalculated_thresholds: Optional pre-calculated thresholds (if None, calculate after bandpass normalization)
        
    Returns:
        tuple: (flags, baseline_info) where flags is a boolean array of the same shape as amp
    """
    # Ensure we have a stream for proper synchronization
    if stream is None:
        stream = cuda.stream()
    
    # Synchronize before starting
    cuda.synchronize()
    
    # Print info
    if baseline_info:
        print(f"Starting SumThreshold on {baseline_info}")
    
    # Set default combinations if not provided
    if combinations is None:
        combinations = [1, 2, 4, 8, 16]

    if rho is None:
        rho = 1.5
    
    print(f"Using window sizes: {combinations}")
    print(f"Using rho value: {rho}")
    
    # Create copies of input data for processing
    processed_amp = amp.copy()
    processed_flags = existing_flags.copy()
    
    # Copy data to GPU
    d_amp = cuda.to_device(processed_amp.astype(np.float32), stream=stream)
    d_flags = cuda.to_device(processed_flags, stream=stream)
    
    # Step 1: Apply bandpass normalization only if thresholds are not pre-calculated
    # When using batch processing, caller should do bandpass normalization before calculating thresholds
    if precalculated_thresholds is None:
        print("Applying bandpass normalization...")
        
        # Launch bandpass normalization kernel (1D grid for channels)
        threads_per_block_1d = 256
        blocks_per_grid_1d = (processed_amp.shape[1] + threads_per_block_1d - 1) // threads_per_block_1d
        
        # Use a fixed RFI threshold of 1.5 (can be adjusted if needed)
        bandpass_normalize_kernel[blocks_per_grid_1d, threads_per_block_1d, stream](
            d_amp, d_flags, 1.5)
        
        # Synchronize after normalization
        stream.synchronize()
        
        # Copy back the normalized data to calculate thresholds
        d_amp.copy_to_host(processed_amp, stream=stream)
        d_flags.copy_to_host(processed_flags, stream=stream)
        
        # Wait for copy to complete
        stream.synchronize()
        
        # Calculate thresholds using the normalized data
        print("Calculating robust channel thresholds...")
        channel_thresholds = calculate_robust_thresholds(processed_amp, processed_flags, sigma_factor)
    else:
        # Use pre-calculated thresholds (in batch processing mode)
        print("Using pre-calculated thresholds")
        channel_thresholds = precalculated_thresholds
    
    # Define thread block and grid dimensions for 2D kernels
    threads_per_block = (16, 16)
    blocks_per_grid_x = (processed_amp.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (processed_amp.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Process each combination size
    start_time = time.time()

    for M in combinations:
        # Calculate thresholds for this combination size
        combo_channel_thresholds = channel_thresholds / (rho ** np.log2(M))

        print(combo_channel_thresholds)
        
        # Copy thresholds to GPU
        d_thresholds = cuda.to_device(combo_channel_thresholds.astype(np.float32), stream=stream)
        
        # Time direction kernel
        sum_threshold_kernel_time_channel[blocks_per_grid, threads_per_block, stream](
            d_amp, d_flags, d_thresholds, M)
        
        # Synchronize between kernels
        stream.synchronize()
        
        # Frequency direction kernel
        sum_threshold_kernel_freq_channel[blocks_per_grid, threads_per_block, stream](
            d_amp, d_flags, d_thresholds, M)
        
        # Synchronize between iterations
        stream.synchronize()
    
    # Final synchronization
    stream.synchronize()
    cuda.synchronize()
    
    # Copy results back to host
    d_flags.copy_to_host(processed_flags, stream=stream)
    
    # Final synchronization
    stream.synchronize()
    cuda.synchronize()
    
    # Process results and generate statistics
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.4f} seconds")
    
    # Calculate percentage of newly flagged data
    existing_count = np.sum(existing_flags)
    total_count = processed_flags.size
    new_flags_count = np.sum(processed_flags) - existing_count
    percent_new_flagged = 100 * new_flags_count / (total_count - existing_count) if (total_count - existing_count) > 0 else 0
    
    print(f"Found {new_flags_count} new flags ({percent_new_flagged:.2f}% of unflagged data) for {baseline_info}")
    # Print shape information for debugging
    print(f"SumThreshold returning flags with shape: {processed_flags.shape}")
    return processed_flags, baseline_info

def get_memory_info():
    """Get information about available GPU and system memory"""
    # Get GPU memory information
    try:
        free_mem, total_mem = cuda.current_context().get_memory_info()
        gpu_usable_mem = free_mem * 0.9  # Use 90% of free memory to be safe
        print(f"GPU memory: {total_mem/1e9:.2f} GB (free: {free_mem/1e9:.2f} GB, usable: {gpu_usable_mem/1e9:.2f} GB)")
    except Exception as e:
        print(f"Warning: Could not determine GPU memory, assuming 8GB: {str(e)}")
        gpu_usable_mem = 6 * 1024 * 1024 * 1024 * 0.8  # Assume 8GB, use 80%
    
    # Get system memory information
    try:
        system_mem = psutil.virtual_memory()
        total_system_mem = system_mem.total
        available_system_mem = system_mem.available
        system_usable_mem = available_system_mem * 0.8  # Use 80% of available memory
        print(f"System memory: {total_system_mem/1e9:.2f} GB (available: {available_system_mem/1e9:.2f} GB, usable: {system_usable_mem/1e9:.2f} GB)")
    except Exception as e:
        print(f"Warning: Could not determine system memory, assuming 16GB: {str(e)}")
        system_usable_mem = 16 * 1024 * 1024 * 1024 * 0.8  # Assume 16GB, use 80%
    
    return gpu_usable_mem, system_usable_mem

def calculate_baseline_batch_size(baseline_data, sample_bl, options, gpu_usable_mem, system_usable_mem):
    """Calculate how many baselines we can process at once based on memory constraints"""
    # Get a sample baseline to estimate memory requirements
    if sample_bl not in baseline_data:
        print("Sample baseline not found in data. Cannot estimate memory requirements.")
        return 1
    
    sample_data = baseline_data[sample_bl]
    
    # Get data dimensions
    time_samples = sample_data.sizes['row']
    freq_channels = sample_data.DATA.shape[1]
    n_corr = sample_data.DATA.shape[2]
    
    # Calculate memory per baseline
    corr_factor = 1  # Default to processing one correlation at a time
    
    # If we're combining correlations, we need to account for that
    if options.get('corr_to_process') and len(options['corr_to_process']) > 1:
        # We're processing multiple correlations together
        corr_factor = len(options['corr_to_process'])
    
    # GPU memory per baseline - account for data, flags, thresholds and intermediate values
    # Use a larger factor for GPU memory to be safe
    gpu_mem_per_bl = time_samples * freq_channels * 4 * 5 * corr_factor  # 4 bytes per float32, 5x for all arrays
    
    # System memory requirements
    system_mem_per_bl = time_samples * freq_channels * 8 * n_corr  # 8 bytes for complex64
    
    # Additional memory for processing
    system_mem_per_bl *= 2  # Account for numpy array copies during processing
    
    # Additional memory for flag application if enabled
    if options.get('apply_flags', False):
        system_mem_per_bl += time_samples * freq_channels * n_corr  # 1 byte per bool flag
    
    # Additional memory for diagnostic plots if enabled
    if options.get('diagnostic_plots', False):
        system_mem_per_bl += time_samples * freq_channels * 4 * 4  # 4x for plot buffers
    
    print(f"Memory estimates per baseline:")
    print(f"  - GPU: {gpu_mem_per_bl / (1024*1024):.2f} MB")
    print(f"  - System: {system_mem_per_bl / (1024*1024):.2f} MB")
    
    # Calculate how many baselines can fit in GPU memory
    gpu_bl_limit = max(1, int(gpu_usable_mem // gpu_mem_per_bl))
    
    # Calculate how many baselines can fit in system memory
    system_bl_limit = max(1, int(system_usable_mem // system_mem_per_bl))
    
    # Take the minimum of the two limits
    bl_batch_size = min(gpu_bl_limit, system_bl_limit)
    
    print(f"Based on memory constraints:")
    print(f"  - GPU limit: {gpu_bl_limit} baselines")
    print(f"  - System memory limit: {system_bl_limit} baselines")
    print(f"  - Selected batch size: {bl_batch_size} baselines")
    
    # # Add a fixed upper limit to be safe
    # max_batch_size = 20
    # if bl_batch_size > max_batch_size:
    #     print(f"Limiting batch size to {max_batch_size} for safety")
    #     bl_batch_size = max_batch_size
    
    return bl_batch_size

def normalize_bandpass_with_polynomial_fit(bandpass, polynomial_degree=5, deviation_threshold=5.0):
    """
    Fit a polynomial to bandpass and detect RFI using absolute residuals.
    
    Args:
        bandpass: 1D array containing bandpass values
        polynomial_degree: Degree of polynomial to fit
        deviation_threshold: Threshold in sigma for flagging outliers
        
    Returns:
        tuple: (smooth_bandpass, rfi_mask, smooth_valid)
    """
    import numpy as np
    from scipy import signal
    
    # Create frequency axis for polynomial fitting
    freq_axis = np.arange(len(bandpass))
    
    # Find channels with valid bandpass values (above zero)
    valid_channels = bandpass > 0
    
    # Initialize outputs
    smooth_bandpass = np.ones_like(bandpass)
    rfi_mask = np.zeros_like(bandpass, dtype=bool)
    smooth_valid = False
    
    if np.sum(valid_channels) < polynomial_degree + 2:
        # Not enough valid channels for polynomial fitting
        print("Warning: Not enough valid channels for polynomial fitting. Using median filter.")
        try:
            smooth_bandpass = signal.medfilt(bandpass, kernel_size=15)
            smooth_valid = True
        except Exception as e:
            print(f"Median filtering failed: {str(e)}")
            smooth_bandpass = bandpass.copy()
    else:
        try:
            # Pre-flag extremely high values that could skew the fit
            # This helps with those large spikes at channel 0
            temp_valid = valid_channels.copy()
            if np.any(valid_channels):
                median_val = np.median(bandpass[valid_channels])
                mad = np.median(np.abs(bandpass[valid_channels] - median_val))
                extreme_threshold = median_val + 10 * mad * 1.4826  # Higher threshold for pre-flagging
                temp_valid = np.logical_and(temp_valid, bandpass < extreme_threshold)
            
            # Fit polynomial to bandpass (in log space to better capture the curve)
            if np.sum(temp_valid) < polynomial_degree + 2:
                # Fall back to original valid channels if too many were pre-flagged
                temp_valid = valid_channels
                
            log_bandpass = np.log10(np.maximum(bandpass[temp_valid], 1e-10))
            valid_freq_axis = freq_axis[temp_valid]
            
            # Polynomial fit
            poly_coeffs = np.polyfit(valid_freq_axis, log_bandpass, polynomial_degree)
            poly_fit = np.polyval(poly_coeffs, freq_axis)
            
            # Clip to avoid overflow
            poly_fit = np.clip(poly_fit, -30, 30)
            
            # Convert back from log space
            smooth_bandpass = np.power(10, poly_fit)
            smooth_valid = True
        except Exception as e:
            print(f"Polynomial fitting failed: {str(e)}")
            try:
                smooth_bandpass = signal.medfilt(bandpass, kernel_size=15)
                smooth_valid = True
            except:
                smooth_bandpass = bandpass.copy()
    
    if smooth_valid:
        # Calculate residuals in normal space
        residuals = bandpass - smooth_bandpass
        
        # Calculate robust statistics on the valid channels only
        valid_residuals = residuals[valid_channels]
        if len(valid_residuals) > 0:
            median_resid = np.median(valid_residuals)
            mad = np.median(np.abs(valid_residuals - median_resid))
            robust_std = mad * 1.4826  # Convert MAD to equivalent of standard deviation
            
            if robust_std > 0:
                # Flag based on absolute residuals exceeding threshold
                rfi_mask[np.abs(residuals - median_resid) > deviation_threshold * robust_std] = True
    
    return smooth_bandpass, rfi_mask, smooth_valid
    
import concurrent.futures

def process_baselines_batch_gpu(baseline_data, field_id, corr_to_process, options, freq_axis):
    """
    Process a batch of baselines with optimized GPU handling and parallel bandpass normalization.
    
    Args:
        baseline_data: Dictionary mapping baselines to their data objects
        field_id: Field ID being processed
        corr_to_process: List of correlations to process
        options: Dictionary of processing options
        freq_axis: Frequency axis information
        
    Returns:
        dict: Dictionary mapping baselines to processed flag results
    """
    total_start_time = time.time()
    
    # Create a main CUDA stream
    main_stream = cuda.stream()
    cuda.synchronize()  # Synchronize before starting
    
    # Extract baselines to process
    baselines = list(baseline_data.keys())
    print(f"Processing batch of {len(baselines)} baselines with optimized GPU transfers...")
    
    # Results dictionary
    results = {}
    
    # Dictionary to store diagnostic data
    diagnostic_data = {} if options['diagnostic_plots'] else None
    
    # Get polynomial degree for fitting from options or use default
    poly_degree = options.get('poly_degree', 5)
    deviation_threshold = options.get('deviation_threshold', 3.0)
    
    # Pre-calculate thresholds for all baselines
    print("Normalizing and pre-calculating thresholds for all baselines...")
    threshold_arrays = {}
    
    # Use multiple threads for bandpass processing
    max_workers = min(options.get('max_threads', 16), len(baselines))
    print(f"Using {max_workers} threads for parallel bandpass processing")
    
    # Function to process a single baseline for bandpass normalization
    def process_baseline_bandpass(bl):
        bl_data = baseline_data[bl]
        
        # Dictionary to store results
        bl_results = {}
        
        # Process different correlation configurations
        if len(corr_to_process) == 2 and corr_to_process[0] == 0 and corr_to_process[-1] == bl_data.DATA.shape[2] - 1:
            # Combined correlations (Stokes I approximation)
            vis_data_0 = bl_data.DATA[:, :, corr_to_process[0]]
            vis_data_1 = bl_data.DATA[:, :, corr_to_process[1]]
            flags_0 = bl_data.FLAG[:, :, corr_to_process[0]]
            flags_1 = bl_data.FLAG[:, :, corr_to_process[1]]
            
            # For flagged points, use 0 for the sum
            masked_data_0 = np.ma.masked_array(vis_data_0, mask=flags_0).filled(0)
            masked_data_1 = np.ma.masked_array(vis_data_1, mask=flags_1).filled(0)
            
            # Sum the data
            vis_sum = masked_data_0 + masked_data_1
            vis_amp = np.abs(vis_sum)
            
            # Combine existing flags
            existing_flags_combined = np.logical_or(flags_0, flags_1)
            
            # Calculate bandpass (median across time for each channel)
            bandpass = np.zeros(vis_amp.shape[1], dtype=np.float32)
            for chan in range(vis_amp.shape[1]):
                unflagged_data = vis_amp[:, chan][~existing_flags_combined[:, chan]]
                if len(unflagged_data) > 0:
                    bandpass[chan] = np.median(unflagged_data)
                else:
                    bandpass[chan] = 0.0
            
            # Apply bandpass normalization using polynomial fitting
            smooth_bandpass, rfi_channels, fit_valid = normalize_bandpass_with_polynomial_fit(
                bandpass, poly_degree, deviation_threshold)
            
            # Print RFI detection results
            rfi_count = np.sum(rfi_channels)
            print(f"Detected {rfi_count} RFI channels in baseline {bl[0]}-{bl[1]} (combined pols)")
            
            # Apply bandpass normalization and flagging
            normalized_amp = vis_amp.copy()
            normalized_flags = existing_flags_combined.copy()
            
            # Flag RFI channels and normalize others
            for chan in range(vis_amp.shape[1]):
                if rfi_channels[chan]:
                    normalized_flags[:, chan] = True
                elif smooth_bandpass[chan] > 0:
                    # Only normalize unflagged data
                    mask = ~normalized_flags[:, chan]
                    normalized_amp[mask, chan] = normalized_amp[mask, chan] / smooth_bandpass[chan]
            
            # Calculate robust thresholds on the normalized data
            base_thresholds = calculate_robust_thresholds(normalized_amp, normalized_flags, options['sigma_factor'])

            print('***********************')
            print(base_thresholds)
            
            bl_results[bl] = {
                'base_thresholds': base_thresholds,
                'vis_amp': normalized_amp,
                'existing_flags': normalized_flags,
                'mode': 'combined',
                'bandpass': bandpass,
                'smooth_bandpass': smooth_bandpass,
                'rfi_channels': rfi_channels
            }
        else:
            # Process each correlation separately
            for corr_idx in corr_to_process:
                vis_data = bl_data.DATA[:, :, corr_idx]
                existing_flags = bl_data.FLAG[:, :, corr_idx]
                
                # Calculate amplitude
                vis_amp = np.abs(vis_data)
                vis_amp = np.ma.masked_array(vis_amp, mask=existing_flags).filled(0)
                
                # Calculate bandpass (median across time for each channel)
                bandpass = np.zeros(vis_amp.shape[1], dtype=np.float32)
                for chan in range(vis_amp.shape[1]):
                    unflagged_data = vis_amp[:, chan][~existing_flags[:, chan]]
                    if len(unflagged_data) > 0:
                        bandpass[chan] = np.median(unflagged_data)
                    else:
                        bandpass[chan] = 0.0
                
                # Apply bandpass normalization using polynomial fitting
                smooth_bandpass, rfi_channels, fit_valid = normalize_bandpass_with_polynomial_fit(
                    bandpass, poly_degree, deviation_threshold)
                
                # Print RFI detection results
                rfi_count = np.sum(rfi_channels)
                print(f"Detected {rfi_count} RFI channels in baseline {bl[0]}-{bl[1]}, pol {corr_idx}")
                
                # Apply bandpass normalization and flagging
                normalized_amp = vis_amp.copy()
                normalized_flags = existing_flags.copy()
                
                # Flag RFI channels and normalize others
                for chan in range(vis_amp.shape[1]):
                    if rfi_channels[chan]:
                        normalized_flags[:, chan] = True
                    elif smooth_bandpass[chan] > 0:
                        # Only normalize unflagged data
                        mask = ~normalized_flags[:, chan]
                        normalized_amp[mask, chan] = normalized_amp[mask, chan] / smooth_bandpass[chan]
                
                # Calculate robust thresholds on the normalized data
                base_thresholds = calculate_robust_thresholds(normalized_amp, normalized_flags, options['sigma_factor'])
                
                # Create unique key for this baseline-correlation
                baseline_corr_key = (bl[0], bl[1], corr_idx)
                
                bl_results[baseline_corr_key] = {
                    'base_thresholds': base_thresholds,
                    'vis_amp': normalized_amp, 
                    'existing_flags': normalized_flags,
                    'mode': 'single',
                    'corr_idx': corr_idx,
                    'bl': bl,
                    'bandpass': bandpass,
                    'smooth_bandpass': smooth_bandpass,
                    'rfi_channels': rfi_channels
                }
        
        return bl_results
    
    # Process baselines in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all baselines for processing
        future_to_bl = {executor.submit(process_baseline_bandpass, bl): bl for bl in baselines}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_bl):
            bl = future_to_bl[future]
            try:
                # Get results for this baseline
                bl_results = future.result()
                # Add to threshold_arrays
                threshold_arrays.update(bl_results)
            except Exception as e:
                print(f"Error processing baseline {bl}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Process each baseline/correlation with dedicated CUDA streams
    # Use a pool of streams for parallel execution
    num_streams = min(5, len(threshold_arrays))  # Limit number of streams
    streams = [cuda.stream() for _ in range(num_streams)]
    
    # Distribute work among streams
    stream_idx = 0
    for key, threshold_data in threshold_arrays.items():
        # Get the current stream
        stream = streams[stream_idx]
        
        # Get process info
        if threshold_data['mode'] == 'single':
            corr_idx = threshold_data['corr_idx']
            bl = threshold_data['bl']
            info_string = f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Pol {corr_idx}"
        else:
            bl = key
            info_string = f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Combined Pols"
        
        # Get data
        vis_amp = threshold_data['vis_amp']
        existing_flags = threshold_data['existing_flags']
        base_thresholds = threshold_data['base_thresholds']
        
        # Run GPU flagger with its own stream - using NORMALIZED data and precalculated thresholds
        flags, _ = sumthreshold_gpu(
            vis_amp, 
            existing_flags, 
            info_string,
            combinations=options['combinations'],
            sigma_factor=None,  # Not used with precalculated thresholds
            rho=options['rho'],
            diagnostic_plots=False,
            stream=stream,
            precalculated_thresholds=base_thresholds
        )
        
        # Combine with existing flags
        combined_flags = np.logical_or(existing_flags, flags)
        
        # Store diagnostic data if enabled
        if options['diagnostic_plots']:
            if threshold_data['mode'] == 'single':
                corr_idx = threshold_data['corr_idx']
                bl = threshold_data['bl']
                diag_key = (bl, corr_idx)
            else:
                diag_key = (bl, "combined")
            
            # Store diagnostic data
            diagnostic_data[diag_key] = {
                'vis_amp': vis_amp.copy(),
                'combined_flags': combined_flags.copy(),
                'existing_flags': existing_flags.copy(),
                'corr_idx': "combined" if threshold_data['mode'] == 'combined' else corr_idx,
                'bl': bl,
                'field_id': field_id,
                'bandpass': threshold_data['bandpass'],
                'smooth_bandpass': threshold_data['smooth_bandpass'],
                'rfi_channels': threshold_data['rfi_channels']
            }
        
        if threshold_data['mode'] == 'combined':
            # For combined mode, update all correlations
            if bl not in results:
                # Initialize full flag array
                bl_data = baseline_data[bl]
                full_flags = bl_data.FLAG.copy()
                results[bl] = full_flags
            
            # Update all correlations with the combined flags
            # Ensure combined_flags is properly broadcast to 3D
            for corr_idx in corr_to_process:
                if len(combined_flags.shape) == 2:
                    # If combined_flags is 2D, we need to broadcast correctly
                    results[bl][:, :, corr_idx] = np.logical_or(
                        results[bl][:, :, corr_idx], 
                        combined_flags)  # NumPy will auto-broadcast here
                else:
                    # If somehow combined_flags is already 3D
                    results[bl][:, :, corr_idx] = np.logical_or(
                        results[bl][:, :, corr_idx], 
                        combined_flags)
        else:
            # Single correlation mode
            corr_idx = threshold_data['corr_idx']
            orig_bl = threshold_data['bl']
            
            if orig_bl not in results:
                # Initialize full flag array
                bl_data = baseline_data[orig_bl]
                full_flags = bl_data.FLAG.copy()
                results[orig_bl] = full_flags
            
            # Update this correlation's flags
            results[orig_bl][:, :, corr_idx] = np.logical_or(
                results[orig_bl][:, :, corr_idx], combined_flags)
        
        # Move to next stream
        stream_idx = (stream_idx + 1) % num_streams
    
    # Synchronize all streams
    for stream in streams:
        stream.synchronize()
    
    # Final synchronization
    cuda.synchronize()

    # Generate diagnostic plots after processing
    if options['diagnostic_plots'] and diagnostic_data:
        print(f"Generating {len(diagnostic_data)} diagnostic plots...")
        output_dir = options.get('output_dir', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        for (bl, corr_idx), data in diagnostic_data.items():
            generate_diagnostic_plot(
                data['vis_amp'], 
                data['combined_flags'], 
                data['existing_flags'], 
                bl, 
                data['field_id'], 
                corr_idx, 
                freq_axis, 
                output_dir
            )
    
    total_time = time.time() - total_start_time
    print(f"Batch processing completed in {total_time:.2f} seconds")
    
    return results

def generate_diagnostic_plot(vis_amp, combined_flags, existing_flags, bl, field_id, corr_idx, freq_axis, output_dir):
    """Generate a diagnostic plot showing flagged and unflagged data"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert corr_idx to string representation
        corr_str = str(corr_idx) if isinstance(corr_idx, (int, float)) else corr_idx
        
        # Create filename
        filename = os.path.join(output_dir, f"pooh_field{field_id}_bl{bl[0]}-{bl[1]}_pol{corr_str}.png")
        
        # For large arrays, subsample to avoid memory issues
        max_size = 1000  # Maximum size for each dimension
        
        time_skip = max(1, vis_amp.shape[0] // max_size)
        freq_skip = max(1, vis_amp.shape[1] // max_size)
        
        # Subsample the arrays
        vis_amp_plot = vis_amp[::time_skip, ::freq_skip]
        combined_flags_plot = combined_flags[::time_skip, ::freq_skip]
        
        # Create figure
        plt.figure(figsize=(12, 6), dpi=80)
        
        # Calculate colormap limits robustly
        valid_data = vis_amp_plot[~combined_flags_plot]
        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, 5)
            vmax = np.percentile(valid_data, 95)
        else:
            vmin, vmax = 0, 1
        
        # Left panel: Raw amplitude
        plt.subplot(1, 2, 1)
        plt.imshow(vis_amp_plot, aspect='auto', origin='lower', cmap='viridis',
                  interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Amplitude', shrink=0.8)
        plt.title('Raw Dynamic Spectra')
        plt.xlabel('Frequency')
        plt.ylabel('Time')
        
        # Right panel: Flagged spectra with all flags in white
        plt.subplot(1, 2, 2)
        
        # Plot the original data first
        plt.imshow(vis_amp_plot, aspect='auto', origin='lower', cmap='viridis',
                   interpolation='nearest', vmin=vmin, vmax=vmax)
        
        # Create a white mask for all flags
        flag_img = np.zeros(vis_amp_plot.shape + (4,), dtype=np.float32)  # RGBA
        
        # Set all flags to white (1, 1, 1)
        flag_img[combined_flags_plot, 0] = 1.0    # R
        flag_img[combined_flags_plot, 1] = 1.0    # G
        flag_img[combined_flags_plot, 2] = 1.0    # B
        flag_img[combined_flags_plot, 3] = 1.0    # Alpha (fully opaque)
        
        # Overlay the white flags
        plt.imshow(flag_img, aspect='auto', origin='lower',
                   interpolation='nearest')
        
        plt.colorbar(label='Amplitude', shrink=0.8)
        plt.title('Flagged Dynamic Spectra')
        plt.xlabel('Frequency')
        plt.ylabel('Time')
        
        # Add simple legend
        import matplotlib.patches as mpatches
        white_patch = mpatches.Patch(color='white', label='Flags')
        plt.legend(handles=[white_patch], loc='upper right')
        
        # Add percentage info for total flags only
        percent_total = 100 * np.sum(combined_flags_plot) / combined_flags_plot.size
        
        plt.suptitle(f'BL {bl[0]}-{bl[1]}, Pol {corr_str}, Field {field_id}\n'
                    f'Total flags: {percent_total:.1f}%')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=80, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error generating diagnostic plot: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')
        gc.collect()

def process_baseline_async(bl_data, bl, field_id, corr_to_process, options, freq_axis, total_flagged, total_visibilities, stream=None, chunk_info="", output_dir='outputs'):
    """Process a single baseline or chunk asynchronously"""
    
    # Initialize time axis
    time_axis = np.arange(bl_data.sizes['row'])
    
    # Initialize flags array if needed
    if options['apply_flags']:
        combined_flags = np.zeros_like(bl_data.FLAG.copy())
    else:
        combined_flags = None
    
    # Track flag counts
    existing_flag_count = 0
    new_flag_count = 0
    
    # Store diagnostic data
    diagnostic_data = []
    
    # Create a stream if not provided
    if stream is None:
        stream = cuda.stream()
    
    # Synchronize to ensure clean GPU state
    cuda.synchronize()
    
    if len(corr_to_process) == 1:
        # Process a single correlation
        corr_idx = corr_to_process[0]
        
        # Extract data
        vis_data = bl_data.DATA[:, :, corr_idx].copy()
        existing_flags = bl_data.FLAG[:, :, corr_idx].copy()
        existing_flag_count += np.sum(existing_flags)
        
        # Calculate amplitude
        masked_data = np.ma.masked_array(vis_data, mask=existing_flags)
        vis_amp = np.abs(masked_data.filled(0))
        
        # Skip if all flagged
        if np.all(existing_flags):
            if options['verbose']:
                print(f"All data already flagged for baseline {bl} corr {corr_idx} {chunk_info}. Skipping.")
            if options['apply_flags']:
                combined_flags[:, :, corr_idx] = existing_flags
            return combined_flags, existing_flag_count, 0
        
        # Run SumThreshold
        new_flags, _ = sumthreshold_gpu(
            vis_amp, 
            existing_flags,
            f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Pol {corr_idx} {chunk_info}",
            combinations=options['combinations'],
            sigma_factor=options['sigma_factor'],
            rho=options['rho'],
            diagnostic_plots=False,
            stream=stream
        )
        
        # Combine flags
        combined_flags_corr = np.logical_or(existing_flags, new_flags)
        
        # Count new flags
        new_flag_count += np.sum(combined_flags_corr) - np.sum(existing_flags)
        
        # Store diagnostic data
        if options['diagnostic_plots']:
            diagnostic_data.append({
                'vis_amp': vis_amp.copy(),
                'combined_flags': combined_flags_corr.copy(),
                'existing_flags': existing_flags.copy(),
                'corr_idx': corr_idx
            })
        
        # Update total visibilities
        total_visibilities += vis_amp.size
        
        # Update combined flags
        if options['apply_flags']:
            combined_flags[:, :, corr_idx] = combined_flags_corr
            
    elif len(corr_to_process) == 2 and corr_to_process[0] == 0 and corr_to_process[-1] == bl_data.DATA.shape[2] - 1:
        # Process combined correlations (Stokes I approximation)
        vis_data_0 = bl_data.DATA[:, :, corr_to_process[0]].copy()
        vis_data_1 = bl_data.DATA[:, :, corr_to_process[1]].copy()
        existing_flags_0 = bl_data.FLAG[:, :, corr_to_process[0]].copy()
        existing_flags_1 = bl_data.FLAG[:, :, corr_to_process[1]].copy()
        
        # Count existing flags
        existing_flag_count += np.sum(existing_flags_0) + np.sum(existing_flags_1)
        
        # Replace flagged values with zeros
        masked_data_0 = np.ma.masked_array(vis_data_0, mask=existing_flags_0).filled(0)
        masked_data_1 = np.ma.masked_array(vis_data_1, mask=existing_flags_1).filled(0)
        
        # Sum data
        vis_sum = masked_data_0 + masked_data_1
        vis_amp = np.abs(vis_sum)
        
        # Combine existing flags
        existing_flags_combined = np.logical_or(existing_flags_0, existing_flags_1)
        
        # Skip if all flagged
        if np.all(existing_flags_combined):
            if options['verbose']:
                print(f"All data already flagged for baseline {bl} combined corrs {chunk_info}. Skipping.")
            if options['apply_flags']:
                for corr_idx in corr_to_process:
                    combined_flags[:, :, corr_idx] = bl_data.FLAG[:, :, corr_idx].copy()
            return combined_flags, existing_flag_count, 0
        
        # Run SumThreshold
        new_flags, _ = sumthreshold_gpu(
            vis_amp, 
            existing_flags_combined,
            f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Combined Pols {chunk_info}",
            combinations=options['combinations'],
            sigma_factor=options['sigma_factor'],
            rho=options['rho'],
            diagnostic_plots=False,
            stream=stream
        )
        
        # Combine flags
        combined_flags_corr = np.logical_or(existing_flags_combined, new_flags)
        
        # Count new flags
        new_flag_count += (np.sum(combined_flags_corr) - np.sum(existing_flags_combined)) * len(corr_to_process)
        
        # Store diagnostic data
        if options['diagnostic_plots']:
            diagnostic_data.append({
                'vis_amp': vis_amp.copy(),
                'combined_flags': combined_flags_corr.copy(),
                'existing_flags': existing_flags_combined.copy(),
                'corr_idx': "combined"
            })
        
        # Update visibilities
        total_visibilities += vis_amp.size * len(corr_to_process)
        
        # Store flags
        if options['apply_flags']:
            for corr_idx in corr_to_process:
                combined_flags[:, :, corr_idx] = combined_flags_corr
    else:
        # Process correlations separately
        for corr_idx in corr_to_process:
            # Extract data
            vis_data = bl_data.DATA[:, :, corr_idx].copy()
            existing_flags = bl_data.FLAG[:, :, corr_idx].copy()
            existing_flag_count += np.sum(existing_flags)
            
            # Calculate amplitude
            masked_data = np.ma.masked_array(vis_data, mask=existing_flags)
            vis_amp = np.abs(masked_data.filled(0))
            
            # Skip if all flagged
            if np.all(existing_flags):
                if options['verbose']:
                    print(f"All data already flagged for baseline {bl} corr {corr_idx} {chunk_info}. Skipping.")
                if options['apply_flags']:
                    combined_flags[:, :, corr_idx] = existing_flags
                continue
            
            # Run SumThreshold
            new_flags, _ = sumthreshold_gpu(
                vis_amp, 
                existing_flags,
                f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Pol {corr_idx} {chunk_info}",
                combinations=options['combinations'],
                sigma_factor=options['sigma_factor'],
                rho=options['rho'],
                diagnostic_plots=False,
                stream=stream
            )
            
            # Combine flags
            combined_flags_corr = np.logical_or(existing_flags, new_flags)
            
            # Count new flags
            new_flag_count += np.sum(combined_flags_corr) - np.sum(existing_flags)
            
            # Store diagnostic data
            if options['diagnostic_plots']:
                diagnostic_data.append({
                    'vis_amp': vis_amp.copy(),
                    'combined_flags': combined_flags_corr.copy(),
                    'existing_flags': existing_flags.copy(),
                    'corr_idx': corr_idx
                })
            
            # Update visibilities
            total_visibilities += vis_amp.size
            
            # Store flags
            if options['apply_flags']:
                combined_flags[:, :, corr_idx] = combined_flags_corr
    
    # Generate diagnostic plots
    if options['diagnostic_plots'] and diagnostic_data:
        os.makedirs(output_dir, exist_ok=True)
        for data in diagnostic_data:
            generate_diagnostic_plot(
                data['vis_amp'], 
                data['combined_flags'], 
                data['existing_flags'], 
                bl, 
                field_id, 
                data['corr_idx'], 
                freq_axis, 
                output_dir
            )
    
    # Ensure synchronization before returning
    if stream:
        stream.synchronize()
    cuda.synchronize()
    
    return combined_flags, existing_flag_count, new_flag_count

def hunt_ms(ms_file, options):
    """The main function which processes the MS file with SumThreshold flagging"""
    print(f"\nPOOH is on the hunt in {ms_file}...")
    
    # Create output directory if needed
    if options['diagnostic_plots']:
        output_dir = options.get('output_dir', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
    
    # Get antenna info
    antenna_ds = xds_from_table(f"{ms_file}::ANTENNA")[0]
    print("\n==== ANTENNA TABLE INFO ====")
    print(f"Number of antennas: {antenna_ds.sizes['row']}")
    print(f"Antenna names: {antenna_ds.NAME.values}")
    
    # Get spectral window info
    spw_ds = xds_from_table(f"{ms_file}::SPECTRAL_WINDOW")[0]
    print("\n==== SPECTRAL WINDOW INFO ====")
    print(f"Number of channels: {spw_ds.NUM_CHAN.values[0]}")
    print(f"Channel frequencies: {spw_ds.CHAN_FREQ.values[0][:5]}... (showing first 5)")
    print(f"Channel width: {spw_ds.CHAN_WIDTH.values[0][0]} Hz")
    
    # Get frequency axis
    freq_axis = spw_ds.CHAN_FREQ.values[0]
    
    # Get memory information
    gpu_usable_mem, system_usable_mem = get_memory_info()
    
    # Track statistics
    total_start_time = time.time()
    total_flagged = 0
    total_new_flags = 0
    total_visibilities = 0
    baselines_processed = 0
    baselines_skipped = 0
    
    # Get field IDs
    print("Getting field IDs...")
    try:
        field_ds = xds_from_table(f"{ms_file}::FIELD")[0]
        field_ids = np.arange(field_ds.sizes['row'])
        print(f"Found field IDs: {field_ids}")
    except Exception as e:
        print(f"Warning: Could not extract field IDs from MS: {str(e)}")
        field_ids = [0]  # Default to field 0
    
    # Process each field
    for field_id in field_ids:
        print(f"\n*** Processing Field {field_id} ***")
        
        # Get unique baselines for this field
        print(f"Getting unique baselines for field {field_id}...")
        baselines = set()
        
        # Read antenna pairs
        taql_where = f"FIELD_ID={field_id}"
        xds_list = xds_from_ms(
            ms_file,
            columns=("ANTENNA1", "ANTENNA2"),
            taql_where=taql_where,
            chunks={"row": 50000}
        )
        
        # Extract unique baselines
        for ds in xds_list:
            ant1, ant2 = dask.compute(ds.ANTENNA1.data, ds.ANTENNA2.data)
            for a1, a2 in zip(ant1, ant2):
                baselines.add((a1, a2))
        
        baselines = list(baselines)
        print(f"Found {len(baselines)} unique baselines in field {field_id}")
        
        # Skip if no baselines
        if not baselines:
            print(f"No baselines found for field {field_id}. Skipping.")
            continue
        
        # Get a sample baseline to check data shape
        sample_bl = baselines[0]
        sample_taql = f"FIELD_ID={field_id} AND ANTENNA1={sample_bl[0]} AND ANTENNA2={sample_bl[1]}"
        
        try:
            sample_ds = xds_from_ms(ms_file, columns=("DATA",), taql_where=sample_taql)[0]
            
            if sample_ds.sizes['row'] == 0:
                print(f"No data found for sample baseline {sample_bl}. Skipping field.")
                continue
                
            data_shape = sample_ds.DATA.shape
        except Exception as e:
            print(f"Error reading sample baseline: {str(e)}")
            try:
                # Try simpler read
                simple_ds = xds_from_ms(ms_file, columns=("DATA",))[0]
                data_shape = simple_ds.DATA.shape
            except Exception as e2:
                print(f"Could not determine data shape: {str(e2)}")
                continue
        
        print(f"Data shape: {data_shape}")
        
        # Get number of correlations
        n_corr = data_shape[2]
        print(f"Number of correlations: {n_corr}")
        
        # Determine correlations to process
        corr_to_process = options['corr_to_process']
        if corr_to_process is None:
            # Default to first and last for Stokes I
            if n_corr >= 2:
                corr_to_process = [0, n_corr-1]
                print(f"Processing correlations 0 and {n_corr-1} (default)")
            else:
                corr_to_process = [0]
                print("Processing correlation 0 (default)")
        else:
            print(f"Processing correlations: {corr_to_process}")
        
        # Calculate ideal batch size
        time_samples = sample_ds.sizes['row']
        freq_channels = data_shape[1]
        
        print(f"Typical baseline has {time_samples} time samples × {freq_channels} frequency channels")
        
        # Process baselines in batches where possible
        baseline_data = {}  # Dictionary to store baseline data
        
        # Try batch processing first
        try:
            # Calculate batch size
            baseline_data[sample_bl] = sample_ds  # Add sample for calculation
            bl_per_batch = calculate_baseline_batch_size(
                baseline_data, sample_bl, options, gpu_usable_mem, system_usable_mem)
            
            if bl_per_batch > 1:
                # Batch processing is possible
                for i in range(0, len(baselines), bl_per_batch):
                    batch = baselines[i:i+bl_per_batch]
                    batch_size = len(batch)
                    
                    print(f"\nProcessing batch {i//bl_per_batch + 1}/{(len(baselines) + bl_per_batch - 1)//bl_per_batch}: {batch_size} baselines")
                    
                    # Read all baselines in one query
                    baseline_clauses = []
                    for ant1, ant2 in batch:
                        baseline_clauses.append(f"(ANTENNA1={ant1} AND ANTENNA2={ant2})")
                    
                    taql_where = f"FIELD_ID={field_id} AND ({' OR '.join(baseline_clauses)})"
                    
                    # Read data
                    batch_ds_list = xds_from_ms(
                        ms_file,
                        columns=("DATA", "FLAG", "ANTENNA1", "ANTENNA2"),
                        taql_where=taql_where
                    )
                    
                    # Dictionary to store baseline data
                    baseline_data = {}
                    
                    # Process each dataset
                    for ds in batch_ds_list:
                        # Materialize data in one operation
                        ant1, ant2, data, flags = dask.compute(
                            ds.ANTENNA1.data, 
                            ds.ANTENNA2.data,
                            ds.DATA.data,
                            ds.FLAG.data
                        )
                        
                        # Group by baseline
                        for i, (a1, a2) in enumerate(zip(ant1, ant2)):
                            bl = (a1, a2)
                            
                            if bl not in batch:
                                continue
                            
                            if bl not in baseline_data:
                                baseline_data[bl] = {
                                    'data': [],
                                    'flags': [],
                                    'indices': []
                                }
                            
                            baseline_data[bl]['data'].append(data[i])
                            baseline_data[bl]['flags'].append(flags[i])
                            baseline_data[bl]['indices'].append(i)
                    
                    # Create data objects
                    valid_baseline_data = {}
                    
                    for bl in batch:
                        if bl not in baseline_data or not baseline_data[bl]['data']:
                            if options['verbose']:
                                print(f"No data found for baseline {bl}. Skipping.")
                            baselines_skipped += 1
                            continue
                        
                        # Combine data
                        bl_data = np.array(baseline_data[bl]['data'])
                        bl_flags = np.array(baseline_data[bl]['flags'])
                        
                        # Check if completely flagged
                        if np.all(bl_flags):
                            if options['verbose']:
                                print(f"Baseline {bl} is completely flagged. Skipping.")
                            baselines_skipped += 1
                            continue
                        
                        # Create data object
                        class MaterializedData:
                            def __init__(self, data, flags, row_size):
                                self.DATA = data
                                self.FLAG = flags
                                self.sizes = {'row': row_size}
                        
                        materialized_bl_data = MaterializedData(
                            bl_data,
                            bl_flags,
                            len(bl_data)
                        )
                        
                        valid_baseline_data[bl] = materialized_bl_data
                    
                    # Process batch on GPU
                    if valid_baseline_data:
                        # Process batch with optimized GPU code
                        batch_results = process_baselines_batch_gpu(
                            valid_baseline_data, 
                            field_id, 
                            corr_to_process, 
                            options, 
                            freq_axis
                        )
                        
                        # Process results
                        for bl, flags in batch_results.items():
                            try:
                                # Get original flags
                                original_flags = valid_baseline_data[bl].FLAG
                                
                                # Count flags
                                existing_count = np.sum(original_flags)
                                total_flags = np.sum(flags)
                                new_count = total_flags - existing_count
                                
                                # Update statistics
                                total_flagged += existing_count
                                total_new_flags += new_count
                                total_visibilities += np.prod(original_flags.shape)
                                
                                baselines_processed += 1
                            except Exception as e:
                                print(f"Error processing results for baseline {bl}: {str(e)}")
                                if options['verbose']:
                                    import traceback
                                    traceback.print_exc()
                                continue
                        
                    # Write flags if requested
                    if options['apply_flags']:
                        try:
                            # Process each baseline separately
                            for bl in batch_results.keys():
                                print(f"Writing flags for baseline {bl}...")
                                
                                # Create baseline-specific query
                                bl_taql = f"FIELD_ID={field_id} AND ANTENNA1={bl[0]} AND ANTENNA2={bl[1]}"
                                
                                # Read original data
                                orig_ds_list = xds_from_ms(
                                    ms_file,
                                    columns=("FLAG",),
                                    taql_where=bl_taql,
                                )
                                
                                if not orig_ds_list or orig_ds_list[0].sizes['row'] == 0:
                                    print(f"No data found for baseline {bl}. Skipping.")
                                    continue
                                    
                                # Get original dataset and flags
                                orig_ds = orig_ds_list[0]
                                orig_flags = orig_ds.FLAG.data.compute()
                                
                                # Get new flags
                                bl_flags = batch_results[bl]
                                
                                # Handle broadcasting from 2D to 3D if needed
                                if len(orig_flags.shape) == 3 and len(bl_flags.shape) == 2:
                                    # Create output with same shape as original flags
                                    combined_flags = orig_flags.copy()
                                    
                                    # Apply new flags to ALL correlations, not just the ones in corr_to_process
                                    for corr_idx in range(orig_flags.shape[2]):
                                        combined_flags[:, :, corr_idx] = np.logical_or(
                                            orig_flags[:, :, corr_idx], 
                                            bl_flags
                                        )
                                else:
                                    # If we already have 3D flags from processing specific correlations
                                    # Make sure all correlations get the flags
                                    if len(bl_flags.shape) == 3 and bl_flags.shape[2] < orig_flags.shape[2]:
                                        combined_flags = orig_flags.copy()
                                        
                                        # Apply processed correlation flags to all correlations
                                        processed_flags = np.any(bl_flags, axis=2, keepdims=True)
                                        for corr_idx in range(orig_flags.shape[2]):
                                            combined_flags[:, :, corr_idx] = np.logical_or(
                                                orig_flags[:, :, corr_idx], 
                                                processed_flags[:, :, 0]
                                            )
                                    else:
                                        # If shapes already match
                                        combined_flags = np.logical_or(orig_flags, bl_flags)
                                
                                # Convert back to dask array with SAME chunking
                                new_flags_dask = da.from_array(combined_flags, chunks=orig_ds.FLAG.data.chunks)
                                
                                # Create updated dataset
                                updated_ds = orig_ds.assign(FLAG=(orig_ds.FLAG.dims, new_flags_dask))
                                
                                # Write back
                                write_back = xds_to_table([updated_ds], ms_file, ["FLAG"])
                                dask.compute(write_back)
                                
                                # Clean up
                                orig_ds_list = None
                                orig_ds = None
                                orig_flags = None
                                combined_flags = None
                                new_flags_dask = None
                                updated_ds = None
                                gc.collect()
                                
                        except Exception as e:
                            print(f"Error writing flags: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print("No valid baselines found in this batch. Skipping.")
                    
                    # Clean up
                    batch_ds_list = None
                    baseline_data = None
                    valid_baseline_data = None
                    batch_results = None
                    
                    # Force gc
                    gc.collect()
                    
                    # Ensure GPU is cleared
                    cuda.synchronize()
            else:
                # Single baseline processing
                raise ValueError("Forcing single baseline processing")
        except Exception as e:
            # Fall back to single baseline processing
            print(f"Batch processing failed: {str(e)}. Falling back to single baseline processing.")
            
            # Process each baseline
            for bl_idx, bl in enumerate(tqdm(baselines, desc="Baselines")):
                if options['verbose']:
                    print(f"\nProcessing baseline {bl_idx+1}/{len(baselines)}: Antennas {bl[0]}-{bl[1]}")
                
                try:
                    # Read baseline data
                    taql_where = f"FIELD_ID={field_id} AND ANTENNA1={bl[0]} AND ANTENNA2={bl[1]}"
                    
                    bl_ds_list = xds_from_ms(
                        ms_file,
                        columns=("DATA", "FLAG"),
                        taql_where=taql_where
                    )
                    
                    # Skip if no data
                    if not bl_ds_list or bl_ds_list[0].sizes['row'] == 0:
                        if options['verbose']:
                            print(f"No data found for baseline {bl}. Skipping.")
                        baselines_skipped += 1
                        continue
                    
                    # Get data
                    bl_data, bl_flags = dask.compute(
                        bl_ds_list[0].DATA.data,
                        bl_ds_list[0].FLAG.data
                    )
                    
                    # Skip if all flagged
                    if np.all(bl_flags):
                        if options['verbose']:
                            print(f"Baseline {bl} is completely flagged. Skipping.")
                        baselines_skipped += 1
                        continue
                    
                    # Create data object
                    class MaterializedData:
                        def __init__(self, data, flags, row_size):
                            self.DATA = data
                            self.FLAG = flags
                            self.sizes = {'row': row_size}
                    
                    materialized_bl_data = MaterializedData(
                        bl_data,
                        bl_flags,
                        bl_data.shape[0]
                    )
                    
                    # Process this baseline
                    # Create a dedicated stream
                    stream = cuda.stream()
                    
                    # Process baseline
                    all_flags, baseline_existing, baseline_new = process_baseline_async(
                        materialized_bl_data, bl, field_id, corr_to_process, options, 
                        freq_axis, total_flagged, total_visibilities, stream,
                        output_dir=options.get('output_dir', 'outputs')
                    )
                    
                    # Update statistics
                    total_flagged += baseline_existing
                    total_new_flags += baseline_new
                    total_visibilities += np.prod(bl_data.shape)
                    baselines_processed += 1
                    
                    # Write flags if requested
                    if options['apply_flags'] and all_flags is not None:
                        try:
                            # Read original data
                            orig_ds_list = xds_from_ms(
                                ms_file,
                                columns=("FLAG",),
                                taql_where=taql_where
                            )
                            
                            if orig_ds_list:
                                # Get original flags
                                orig_ds = orig_ds_list[0]
                                orig_flags = orig_ds.FLAG
                                
                                # Combine flags
                                orig_flags_array = orig_ds.FLAG.data.compute()
                                if len(orig_flags_array.shape) != len(all_flags.shape):
                                    print(f"Handling dimension mismatch. Original: {orig_flags_array.shape}, New: {all_flags.shape}")
                                    
                                    # Create a properly sized output array
                                    combined_flags = np.copy(orig_flags_array)
                                    
                                    if len(orig_flags_array.shape) == 3 and len(all_flags.shape) == 2:
                                        # If original is 3D and new is 2D
                                        for corr_idx in range(orig_flags_array.shape[2]):
                                            # Apply 2D flags to each correlation plane
                                            combined_flags[:, :, corr_idx] = np.logical_or(
                                                orig_flags_array[:, :, corr_idx], 
                                                all_flags)
                                    else:
                                        print("Unexpected dimension mismatch. Using original flags.")
                                        combined_flags = orig_flags_array
                                else:
                                    # Normal case - same dimensions
                                    combined_flags = np.logical_or(orig_flags_array, all_flags)
                                
                                # Convert back to dask array
                                new_flags_dask = da.from_array(combined_flags, chunks=orig_flags.data.chunks)
                                
                                # Create updated dataset
                                updated_ds = orig_ds.assign(FLAG=(orig_flags.dims, new_flags_dask))
                                
                                # Write back
                                print(f"Writing flags for baseline {bl}...")
                                write_back = xds_to_table([updated_ds], ms_file, ["FLAG"])
                                dask.compute(write_back)
                        except Exception as e:
                            print(f"Error writing flags for baseline {bl}: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    
                    # Clean up
                    stream.synchronize()
                    bl_ds_list = None
                    bl_data = None
                    bl_flags = None
                    materialized_bl_data = None
                    all_flags = None
                    
                    # Force gc
                    gc.collect()
                    
                    # Ensure GPU is clear
                    cuda.synchronize()
                    
                except Exception as e:
                    print(f"Error processing baseline {bl}: {str(e)}")
                    if options['verbose']:
                        import traceback
                        traceback.print_exc()
                    continue
    
    # Calculate final statistics
    total_time = time.time() - total_start_time
    overall_percent_flagged = 100 * (total_flagged + total_new_flags) / total_visibilities if total_visibilities > 0 else 0
    new_percent_flagged = 100 * total_new_flags / total_visibilities if total_visibilities > 0 else 0
    
    # Return results
    results = {
        'total_processing_time': total_time,
        'overall_percent_flagged': overall_percent_flagged,
        'new_percent_flagged': new_percent_flagged,
        'total_visibilities': total_visibilities,
        'existing_flags': total_flagged,
        'new_flags': total_new_flags,
        'baselines_processed': baselines_processed,
        'baselines_skipped': baselines_skipped
    }
    
    return results

# def main():
#     """Main function"""
#     parser = argparse.ArgumentParser(description='POOH: Parallelized Optimized Outlier Hunter')
    
#     # Required arguments
#     parser.add_argument('ms_path', help='Path to the MS file')
    
#     # Optional arguments
#     parser.add_argument('--combinations', type=str, default='1,2,4,8,16,32,64',
#                        help='Comma-separated list of window sizes')
#     parser.add_argument('--sigma', type=float, default=6.0,
#                        help='Sigma factor for threshold calculation')
#     parser.add_argument('--rho', type=float, default=1.5,
#                        help='Factor to reduce threshold for larger window sizes')
#     parser.add_argument('--polarizations', type=str, default=None,
#                        help='Comma-separated list of polarization indices to process (default: all)')
#     parser.add_argument('--apply-flags', action='store_true',
#                        help='Apply flags to the MS file')
#     parser.add_argument('--diagnostic-plots', action='store_true',
#                        help='Generate diagnostic plots')
#     parser.add_argument('--output-dir', type=str, default='outputs',
#                        help='Directory to save diagnostic plots (default: outputs)')
#     parser.add_argument('--max-memory-usage', type=float, default=0.8,
#                        help='Maximum fraction of available memory to use (default: 0.8)')
#     parser.add_argument('--verbose', action='store_true',
#                        help='Enable verbose output')
                       
#     args = parser.parse_args()
    
#     # Print GPU information
#     print_gpu_info()
    
#     # Parse combinations
#     combinations = [int(x) for x in args.combinations.split(',')]
    
#     # Parse polarizations
#     corr_to_process = None
#     if args.polarizations:
#         corr_to_process = [int(x) for x in args.polarizations.split(',')]
    
#     # Prepare options
#     options = {
#         'combinations': combinations,
#         'sigma_factor': args.sigma,
#         'rho': args.rho,
#         'corr_to_process': corr_to_process,
#         'apply_flags': args.apply_flags,
#         'diagnostic_plots': args.diagnostic_plots,
#         'output_dir': args.output_dir,
#         'max_memory_usage': args.max_memory_usage,
#         'verbose': args.verbose
#     }

#     print(" Parrallelized Optimized Outlier Hunter: POOH, Developed by Arpan Pal at NCRA-T