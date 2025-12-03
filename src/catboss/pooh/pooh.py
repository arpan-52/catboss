import gc
import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import dask.array as da
from daskms import xds_from_ms, xds_from_table, xds_to_table
from tqdm import tqdm
import dask
import psutil
import concurrent.futures

# Check GPU availability and import Numba
try:
    from numba import cuda, jit, prange

    GPU_AVAILABLE = cuda.is_available()
    if GPU_AVAILABLE:
        print("GPU detected - will use GPU acceleration")
    else:
        print("No GPU detected - will use CPU processing")
except Exception as e:
    print(f"GPU check failed: {e} - will use CPU processing")
    from numba import jit, prange

    GPU_AVAILABLE = False
    cuda = None


def apply_flags_to_ms(ms_file, bl, field_id, new_flags):
    """Apply flags to the MS file for a specific baseline"""
    # Create baseline-specific query
    bl_taql = f"FIELD_ID={field_id} AND ANTENNA1={bl[0]} AND ANTENNA2={bl[1]}"

    # Read original data
    orig_ds_list = xds_from_ms(ms_file, columns=("FLAG",), taql_where=bl_taql)

    if not orig_ds_list or orig_ds_list[0].sizes["row"] == 0:
        return False

    try:
        # Get original dataset and flags
        orig_ds = orig_ds_list[0]
        orig_flags = orig_ds.FLAG.data.compute()

        # Ensure new_flags matches the shape of orig_flags
        if orig_flags.shape != new_flags.shape:
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
                        new_flags[:min_rows, :min_chans],
                    )
            else:
                # For other dimension mismatches, just use common dimensions
                common_dims = tuple(
                    min(d1, d2) for d1, d2 in zip(orig_flags.shape, new_flags.shape)
                )
                slices = tuple(slice(0, d) for d in common_dims)

                combined_flags[slices] = np.logical_or(
                    orig_flags[slices], new_flags[slices]
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
        write_back = xds_to_table([updated_ds], ms_file, ["FLAG"])
        dask.compute(write_back)

        return True
    except Exception:
        return False


def print_gpu_info():
    """Print basic information about available GPUs or CPU mode"""
    print("=== Hardware Information ===")
    if not GPU_AVAILABLE:
        print("Running in CPU mode - no GPU available")
        print("============================")
        return

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

    print("============================")


# ==================== CPU IMPLEMENTATIONS ====================
@jit(nopython=True)
def sum_threshold_cpu_time_channel(amp, flags, thresholds, M):
    """CPU implementation of SumThreshold in time direction (JIT-compiled for speed)"""
    for i in range(amp.shape[0]):
        for j in range(amp.shape[1] - M + 1):
            # Calculate average threshold for this group
            avg_threshold = np.mean(thresholds[j : j + M])

            # Count unflagged points and calculate their sum
            group_sum = 0.0
            count = 0
            for k in range(M):
                if not flags[i, j + k]:
                    group_sum += amp[i, j + k]
                    count += 1

            # Only proceed if we have enough unflagged points (at least 30%)
            min_unflagged = max(1, int(M * 0.3))

            if count >= min_unflagged:
                # Check if the average exceeds the threshold
                if (group_sum / count) > avg_threshold:
                    # Flag samples that exceed the threshold
                    for k in range(M):
                        if not flags[i, j + k]:
                            # Flag more conservatively for low amplitudes
                            if amp[i, j + k] < avg_threshold * 0.5:
                                # For very low amplitudes, require higher confidence
                                if amp[i, j + k] > thresholds[j + k] * 1.2:
                                    flags[i, j + k] = True
                            else:
                                # For normal/high amplitudes, use standard threshold
                                if amp[i, j + k] > thresholds[j + k]:
                                    flags[i, j + k] = True


@jit(nopython=True)
def sum_threshold_cpu_freq_channel(amp, flags, thresholds, M):
    """CPU implementation of SumThreshold in frequency direction (JIT-compiled for speed)"""
    for i in range(amp.shape[0] - M + 1):
        for j in range(amp.shape[1]):
            # Get threshold for this channel
            threshold = thresholds[j]

            # Count unflagged points and calculate their sum
            group_sum = 0.0
            count = 0
            for k in range(M):
                if not flags[i + k, j]:
                    group_sum += amp[i + k, j]
                    count += 1

            # Only proceed if we have enough unflagged points (at least 30%)
            min_unflagged = max(1, int(M * 0.3))

            if count >= min_unflagged:
                # Check if the average exceeds the threshold
                if (group_sum / count) > threshold:
                    # Flag samples that exceed the threshold
                    for k in range(M):
                        if not flags[i + k, j]:
                            # Flag more conservatively for low amplitudes
                            if amp[i + k, j] < threshold * 0.5:
                                # For very low amplitudes, require higher confidence
                                if amp[i + k, j] > threshold * 1.2:
                                    flags[i + k, j] = True
                            else:
                                # For normal/high amplitudes, use standard threshold
                                if amp[i + k, j] > threshold:
                                    flags[i + k, j] = True


@jit
def bandpass_normalize_cpu(amp, flags, rfi_threshold):
    """CPU implementation of bandpass normalization (JIT-compiled, uses numpy mode)"""
    for channel_idx in range(amp.shape[1]):
        # Calculate median for this channel
        unflagged_data = amp[~flags[:, channel_idx], channel_idx]

        if len(unflagged_data) > 0:
            bandpass_value = np.median(unflagged_data)
        else:
            bandpass_value = 0.0

        # Check if channel is RFI-affected by comparing with neighbors
        is_rfi = False
        if bandpass_value > 0:
            # Calculate neighbor statistics
            neighbor_vals = []
            for offset in range(-3, 4):
                if offset != 0:
                    neighbor_idx = channel_idx + offset
                    if 0 <= neighbor_idx < amp.shape[1]:
                        neighbor_data = amp[~flags[:, neighbor_idx], neighbor_idx]
                        if len(neighbor_data) > 0:
                            neighbor_vals.append(np.median(neighbor_data))

            # Check if this channel is an outlier
            if len(neighbor_vals) > 0:
                neighbor_avg = np.mean(neighbor_vals)
                if bandpass_value > neighbor_avg * rfi_threshold:
                    is_rfi = True

        # Normalize or flag
        if is_rfi:
            # Flag RFI channel
            flags[:, channel_idx] = True
        elif bandpass_value > 0:
            # Normalize by bandpass (only unflagged data)
            mask = ~flags[:, channel_idx]
            amp[mask, channel_idx] = amp[mask, channel_idx] / bandpass_value


# ==================== GPU IMPLEMENTATIONS ====================
if GPU_AVAILABLE:

    @cuda.jit
    def sum_threshold_kernel_time_channel(amp, flags, thresholds, M):
        """Improved CUDA kernel for SumThreshold in time direction with better flagging logic"""
        i, j = cuda.grid(2)

        if i < amp.shape[0] and j < amp.shape[1] - M + 1:
            # Calculate average threshold for this group
            avg_threshold = 0.0
            for k in range(M):
                avg_threshold += thresholds[j + k]
            avg_threshold /= M

            # Count unflagged points and calculate their sum
            group_sum = 0.0
            count = 0

            for k in range(M):
                if not flags[i, j + k]:
                    group_sum += amp[i, j + k]
                    count += 1

            # Only proceed if we have enough unflagged points (at least 30%)
            min_unflagged = max(1, int(M * 0.3))

            if count >= min_unflagged:
                # Check if the average exceeds the threshold
                if (group_sum / count) > avg_threshold:
                    # Flag samples that exceed the threshold
                    for k in range(M):
                        if not flags[i, j + k]:
                            # Flag more conservatively for low amplitudes
                            if amp[i, j + k] < avg_threshold * 0.5:
                                # For very low amplitudes, require higher confidence
                                if amp[i, j + k] > thresholds[j + k] * 1.2:
                                    flags[i, j + k] = True
                            else:
                                # For normal/high amplitudes, use standard threshold
                                if amp[i, j + k] > thresholds[j + k]:
                                    flags[i, j + k] = True

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
                if not flags[i + k, j]:
                    group_sum += amp[i + k, j]
                    count += 1

            # Only proceed if we have enough unflagged points (at least 30%)
            min_unflagged = max(1, int(M * 0.3))

            if count >= min_unflagged:
                # Check if the average exceeds the threshold
                if (group_sum / count) > threshold:
                    # Flag samples that exceed the threshold
                    for k in range(M):
                        if not flags[i + k, j]:
                            # Flag more conservatively for low amplitudes
                            if amp[i + k, j] < threshold * 0.5:
                                # For very low amplitudes, require higher confidence
                                if amp[i + k, j] > threshold * 1.2:
                                    flags[i + k, j] = True
                            else:
                                # For normal/high amplitudes, use standard threshold
                                if amp[i + k, j] > threshold:
                                    flags[i + k, j] = True

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
                    for j in range(i + 1, count):
                        if values[i] > values[j]:
                            values[i], values[j] = values[j], values[i]

                # Get median
                if count % 2 == 0:
                    bandpass_value = (values[count // 2 - 1] + values[count // 2]) / 2
                else:
                    bandpass_value = values[count // 2]

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
                            for t_step in range(
                                0, amp.shape[0], amp.shape[0] // 10 + 1
                            ):
                                if (
                                    t_step < amp.shape[0]
                                    and not flags[t_step, neighbor_idx]
                                    and n_count < 10
                                ):
                                    neighbor_vals[n_count] = amp[t_step, neighbor_idx]
                                    n_count += 1

                            # Get median of samples
                            if n_count > 0:
                                # Sort
                                for i in range(n_count):
                                    for j in range(i + 1, n_count):
                                        if neighbor_vals[i] > neighbor_vals[j]:
                                            neighbor_vals[i], neighbor_vals[j] = (
                                                neighbor_vals[j],
                                                neighbor_vals[i],
                                            )

                                # Add to statistics
                                if n_count % 2 == 0:
                                    neighbor_sum += (
                                        neighbor_vals[n_count // 2 - 1]
                                        + neighbor_vals[n_count // 2]
                                    ) / 2
                                else:
                                    neighbor_sum += neighbor_vals[n_count // 2]
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


@jit(nopython=True, parallel=True)
def calculate_channel_medians_parallel(amp, flags, sigma_factor):
    """
    Optimized parallel calculation of channel medians using Numba.
    Processes all channels in parallel across CPU cores.
    """
    n_time, n_chan = amp.shape
    channel_thresholds = np.zeros(n_chan, dtype=np.float32)

    # Process channels in parallel
    for j in prange(n_chan):
        # Count unflagged points
        n_unflagged = 0
        for i in range(n_time):
            if not flags[i, j]:
                n_unflagged += 1

        if n_unflagged > 0:
            # Allocate array for unflagged data
            unflagged_data = np.empty(n_unflagged, dtype=amp.dtype)
            idx = 0
            for i in range(n_time):
                if not flags[i, j]:
                    unflagged_data[idx] = amp[i, j]
                    idx += 1

            # Calculate median
            median = np.median(unflagged_data)
            channel_thresholds[j] = median * sigma_factor
        else:
            # Mark as needing interpolation
            channel_thresholds[j] = -1.0

    return channel_thresholds


def calculate_robust_thresholds(amp, flags, sigma_factor=6.0):
    """
    Calculate robust thresholds using median - OPTIMIZED VERSION.
    Uses Numba JIT compilation with parallel processing for 5-10x speedup.
    """
    # Use optimized parallel Numba function
    channel_thresholds = calculate_channel_medians_parallel(amp, flags, sigma_factor)

    # Handle fully flagged channels (marked with -1)
    flagged_channels = np.where(channel_thresholds < 0)[0]

    if len(flagged_channels) > 0:
        valid_channels = np.where(channel_thresholds > 0)[0]

        if len(valid_channels) > 0:
            # Interpolate from nearby valid channels
            for j in flagged_channels:
                if len(valid_channels) == 1:
                    channel_thresholds[j] = channel_thresholds[valid_channels[0]]
                else:
                    # Find closest valid channels
                    left_idx = valid_channels[valid_channels < j]
                    right_idx = valid_channels[valid_channels > j]

                    if len(left_idx) > 0 and len(right_idx) > 0:
                        # Interpolate
                        left_val = channel_thresholds[left_idx[-1]]
                        right_val = channel_thresholds[right_idx[0]]
                        channel_thresholds[j] = (left_val + right_val) / 2
                    elif len(left_idx) > 0:
                        channel_thresholds[j] = channel_thresholds[left_idx[-1]]
                    else:
                        channel_thresholds[j] = channel_thresholds[right_idx[0]]
        else:
            # All channels flagged - use global median
            global_median = np.median(amp) * sigma_factor
            channel_thresholds[:] = global_median

    return channel_thresholds


@jit(nopython=True, parallel=True)
def calculate_bandpass_parallel(amp, flags):
    """
    Optimized parallel bandpass calculation using Numba.
    Calculates median across time for each channel in parallel.
    5-10x faster than sequential version.
    """
    n_time, n_chan = amp.shape
    bandpass = np.zeros(n_chan, dtype=np.float32)

    # Process channels in parallel
    for j in prange(n_chan):
        # Count unflagged points
        n_unflagged = 0
        for i in range(n_time):
            if not flags[i, j]:
                n_unflagged += 1

        if n_unflagged > 0:
            # Allocate array for unflagged data
            unflagged_data = np.empty(n_unflagged, dtype=amp.dtype)
            idx = 0
            for i in range(n_time):
                if not flags[i, j]:
                    unflagged_data[idx] = amp[i, j]
                    idx += 1

            # Calculate median
            bandpass[j] = np.median(unflagged_data)
        else:
            bandpass[j] = 0.0

    return bandpass


@jit(nopython=True, parallel=True)
def apply_bandpass_normalization_parallel(amp, flags, smooth_bandpass, rfi_channels):
    """
    Optimized parallel bandpass normalization using Numba.
    Applies normalization and RFI flagging in-place across channels in parallel.
    2-3x faster than sequential version.
    """
    n_time, n_chan = amp.shape

    # Process channels in parallel
    for j in prange(n_chan):
        if rfi_channels[j]:
            # Flag entire RFI channel
            for i in range(n_time):
                flags[i, j] = True
        elif smooth_bandpass[j] > 0:
            # Normalize unflagged data
            for i in range(n_time):
                if not flags[i, j]:
                    amp[i, j] = amp[i, j] / smooth_bandpass[j]


def sumthreshold_gpu(
    amp,
    existing_flags,
    baseline_info,
    combinations=None,
    sigma_factor=6.0,
    rho=1.5,
    diagnostic_plots=False,
    stream=None,
    precalculated_thresholds=None,
    logger=None,
):
    """
    SumThreshold implementation with integrated bandpass normalization
    Now supports both CPU and GPU execution.

    Args:
        amp: Amplitude array (2D: time × frequency)
        existing_flags: Boolean array of existing flags (same shape as amp)
        baseline_info: String description of the baseline for logging
        combinations: List of window sizes to use for SumThreshold
        sigma_factor: Multiplier for standard deviation in threshold calculation (not used with precalculated_thresholds)
        rho: Factor to reduce threshold for larger window sizes
        diagnostic_plots: Whether to generate diagnostic plots
        stream: CUDA stream to use for asynchronous execution (GPU only)
        precalculated_thresholds: Optional pre-calculated thresholds (if None, calculate after bandpass normalization)
        logger: Logger instance for logging messages

    Returns:
        tuple: (flags, baseline_info) where flags is a boolean array of the same shape as amp
    """
    # Handle stream and synchronization only if GPU is available
    if GPU_AVAILABLE:
        # Ensure we have a stream for proper synchronization
        if stream is None:
            stream = cuda.stream()
        # Synchronize before starting
        cuda.synchronize()

    # Log info
    if baseline_info and logger:
        logger.debug(f"Starting SumThreshold on {baseline_info}")

    # Set default combinations if not provided
    if combinations is None:
        combinations = [1, 2, 4, 8, 16]

    if rho is None:
        rho = 1.5

    if logger:
        logger.debug(f"Using window sizes: {combinations}")
        logger.debug(f"Using rho value: {rho}")

    # Create copies of input data for processing
    processed_amp = amp.copy()
    processed_flags = existing_flags.copy()

    # Step 1: Apply bandpass normalization only if thresholds are not pre-calculated
    if precalculated_thresholds is None:
        if logger:
            logger.debug("Applying bandpass normalization...")

        if GPU_AVAILABLE:
            # GPU path: Copy data to GPU
            d_amp = cuda.to_device(processed_amp.astype(np.float32), stream=stream)
            d_flags = cuda.to_device(processed_flags, stream=stream)

            # Launch bandpass normalization kernel (1D grid for channels)
            threads_per_block_1d = 256
            blocks_per_grid_1d = (
                processed_amp.shape[1] + threads_per_block_1d - 1
            ) // threads_per_block_1d

            # Use a fixed RFI threshold of 1.5 (can be adjusted if needed)
            bandpass_normalize_kernel[blocks_per_grid_1d, threads_per_block_1d, stream](
                d_amp, d_flags, 1.5
            )

            # Synchronize after normalization
            stream.synchronize()

            # Copy back the normalized data to calculate thresholds
            d_amp.copy_to_host(processed_amp, stream=stream)
            d_flags.copy_to_host(processed_flags, stream=stream)

            # Wait for copy to complete
            stream.synchronize()
        else:
            # CPU path: Direct normalization
            bandpass_normalize_cpu(processed_amp, processed_flags, 1.5)

        # Calculate thresholds using the normalized data
        if logger:
            logger.debug("Calculating robust channel thresholds...")
        channel_thresholds = calculate_robust_thresholds(
            processed_amp, processed_flags, sigma_factor
        )
    else:
        # Use pre-calculated thresholds (in batch processing mode)
        if logger:
            logger.debug("Using pre-calculated thresholds")
        channel_thresholds = precalculated_thresholds

    # Process each combination size
    start_time = time.time()

    if GPU_AVAILABLE:
        # GPU path: Copy data to GPU if not already there
        if precalculated_thresholds is not None:
            d_amp = cuda.to_device(processed_amp.astype(np.float32), stream=stream)
            d_flags = cuda.to_device(processed_flags, stream=stream)

        # Define thread block and grid dimensions for 2D kernels
        threads_per_block = (16, 16)
        blocks_per_grid_x = (
            processed_amp.shape[0] + threads_per_block[0] - 1
        ) // threads_per_block[0]
        blocks_per_grid_y = (
            processed_amp.shape[1] + threads_per_block[1] - 1
        ) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        for M in combinations:
            # Calculate thresholds for this combination size
            combo_channel_thresholds = channel_thresholds / (rho ** np.log2(M))

            # Copy thresholds to GPU
            d_thresholds = cuda.to_device(
                combo_channel_thresholds.astype(np.float32), stream=stream
            )

            # Time direction kernel
            sum_threshold_kernel_time_channel[
                blocks_per_grid, threads_per_block, stream
            ](d_amp, d_flags, d_thresholds, M)

            # Synchronize between kernels
            stream.synchronize()

            # Frequency direction kernel
            sum_threshold_kernel_freq_channel[
                blocks_per_grid, threads_per_block, stream
            ](d_amp, d_flags, d_thresholds, M)

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
    else:
        # CPU path: Process directly on CPU
        for M in combinations:
            # Calculate thresholds for this combination size
            combo_channel_thresholds = channel_thresholds / (rho ** np.log2(M))

            # Time direction processing
            sum_threshold_cpu_time_channel(
                processed_amp, processed_flags, combo_channel_thresholds, M
            )

            # Frequency direction processing
            sum_threshold_cpu_freq_channel(
                processed_amp, processed_flags, combo_channel_thresholds, M
            )

    # Process results and generate statistics
    processing_time = time.time() - start_time
    if logger:
        logger.debug(f"Processing completed in {processing_time:.4f} seconds")

    # Calculate percentage of newly flagged data
    existing_count = np.sum(existing_flags)
    total_count = processed_flags.size
    new_flags_count = np.sum(processed_flags) - existing_count
    percent_new_flagged = (
        100 * new_flags_count / (total_count - existing_count)
        if (total_count - existing_count) > 0
        else 0
    )

    if logger:
        logger.debug(
            f"Found {new_flags_count} new flags ({percent_new_flagged:.2f}% of unflagged data) for {baseline_info}"
        )
        # Log shape information for debugging
        logger.debug(f"SumThreshold returning flags with shape: {processed_flags.shape}")
    return processed_flags, baseline_info


def get_memory_info(logger=None):
    """Get information about available GPU and system memory"""
    # Get GPU memory information
    if GPU_AVAILABLE:
        try:
            free_mem, total_mem = cuda.current_context().get_memory_info()
            gpu_usable_mem = free_mem * 0.9  # Use 90% of free memory to be safe
            logger.info(
                f"   GPU: {total_mem / 1e9:.2f} GB total | {free_mem / 1e9:.2f} GB free | {gpu_usable_mem / 1e9:.2f} GB usable"
            )
        except Exception as e:
            logger.warning(
                f"Could not determine GPU memory, assuming 8GB: {str(e)}"
            )
            gpu_usable_mem = 6 * 1024 * 1024 * 1024 * 0.8  # Assume 8GB, use 80%
    else:
        gpu_usable_mem = 0
        logger.info("   No GPU available - using CPU only")

    # Get system memory information
    try:
        system_mem = psutil.virtual_memory()
        total_system_mem = system_mem.total
        available_system_mem = system_mem.available
        system_usable_mem = (
            available_system_mem * 0.6
        )  # Use 60% of available memory (conservative)
        logger.info(
            f"   RAM: {total_system_mem / 1e9:.2f} GB total | {available_system_mem / 1e9:.2f} GB available | {system_usable_mem / 1e9:.2f} GB usable"
        )
    except Exception as e:
        logger.warning(
            f"Could not determine system memory, assuming 16GB: {str(e)}"
        )
        system_usable_mem = 16 * 1024 * 1024 * 1024 * 0.6  # Assume 16GB, use 60%

    return gpu_usable_mem, system_usable_mem


def estimate_field_memory_requirements(ms_file, field_id, sample_baseline_count=5, logger=None):
    """
    Estimate memory requirements for processing a single field.
    Samples a few baselines to get accurate estimates.
    """
    try:
        # Get a few sample baselines to estimate data size
        taql_where = f"FIELD_ID={field_id}"
        sample_ds_list = xds_from_ms(
            ms_file,
            columns=("DATA", "FLAG", "ANTENNA1", "ANTENNA2"),
            taql_where=taql_where,
            chunks={"row": 10000},
        )

        if not sample_ds_list:
            return 0

        sample_ds = sample_ds_list[0]

        # Get dimensions
        n_rows = sample_ds.sizes["row"]
        data_shape = sample_ds.DATA.shape
        n_chan = data_shape[1]
        n_corr = data_shape[2]

        # Estimate per-baseline size
        # Data: complex64 (8 bytes) + Flags: bool (1 byte) + working memory (4x multiplier for copies/temp arrays)
        bytes_per_sample = (8 + 1) * n_corr * 4  # 4x multiplier for safety
        total_bytes = n_rows * n_chan * bytes_per_sample

        if logger:
            logger.info(
                f"Field {field_id} memory estimate: {n_rows} rows × {n_chan} channels × {n_corr} corr = {total_bytes / 1e9:.2f} GB"
            )

        return total_bytes

    except Exception as e:
        if logger:
            logger.warning(f"Warning: Could not estimate memory for field {field_id}: {str(e)}")
        # Conservative estimate: 30GB per field
        return 30 * 1024 * 1024 * 1024


def determine_parallel_field_count(ms_file, field_ids, available_memory, logger=None):
    """
    Determine how many fields can be processed in parallel based on available memory.
    """
    if len(field_ids) == 0:
        return 1

    # Estimate memory for first field (assume all fields are similar size)
    field_mem = estimate_field_memory_requirements(ms_file, field_ids[0], logger=logger)

    if field_mem == 0:
        return 1

    # Calculate how many fields can fit
    max_parallel_fields = max(1, int(available_memory / field_mem))

    # Cap at number of available fields
    max_parallel_fields = min(max_parallel_fields, len(field_ids))

    # Cap at reasonable maximum (8 fields) to avoid too many processes
    max_parallel_fields = min(max_parallel_fields, 8)

    if logger:
        logger.info(f"\n{'=' * 60}")
        logger.info("ADAPTIVE PARALLELIZATION CONFIGURATION")
        logger.info(f"{'=' * 60}")
        logger.info(f"Available RAM: {available_memory / 1e9:.2f} GB")
        logger.info(f"Estimated memory per field: {field_mem / 1e9:.2f} GB")
        logger.info(f"Total fields to process: {len(field_ids)}")
        logger.info(f"Fields that will be processed in parallel: {max_parallel_fields}")
        logger.info(f"{'=' * 60}\n")

    return max_parallel_fields


def calculate_baseline_batch_size(
    baseline_data, sample_bl, options, gpu_usable_mem, system_usable_mem
):
    """Calculate how many baselines we can process at once based on memory constraints"""
    # Get logger from options
    logger = options.get("logger")

    # Get a sample baseline to estimate memory requirements
    if sample_bl not in baseline_data:
        logger.warning("Sample baseline not found in data. Cannot estimate memory requirements.")
        return 1

    sample_data = baseline_data[sample_bl]

    # Get data dimensions
    time_samples = sample_data.sizes["row"]
    freq_channels = sample_data.DATA.shape[1]
    n_corr = sample_data.DATA.shape[2]

    # Calculate memory per baseline
    corr_factor = 1  # Default to processing one correlation at a time

    # If we're combining correlations, we need to account for that
    if options.get("corr_to_process") and len(options["corr_to_process"]) > 1:
        # We're processing multiple correlations together
        corr_factor = len(options["corr_to_process"])

    # GPU memory per baseline - account for data, flags, thresholds and intermediate values
    # Use a larger factor for GPU memory to be safe
    gpu_mem_per_bl = (
        time_samples * freq_channels * 4 * 5 * corr_factor
    )  # 4 bytes per float32, 5x for all arrays

    # System memory requirements
    system_mem_per_bl = (
        time_samples * freq_channels * 8 * n_corr
    )  # 8 bytes for complex64

    # Additional memory for processing
    system_mem_per_bl *= 2  # Account for numpy array copies during processing

    # Additional memory for flag application if enabled
    if options.get("apply_flags", False):
        system_mem_per_bl += (
            time_samples * freq_channels * n_corr
        )  # 1 byte per bool flag

    # Additional memory for diagnostic plots if enabled
    if options.get("diagnostic_plots", False):
        system_mem_per_bl += time_samples * freq_channels * 4 * 4  # 4x for plot buffers

    logger.info("Memory estimates per baseline:")
    logger.info(f"  - GPU: {gpu_mem_per_bl / (1024 * 1024):.2f} MB")
    logger.info(f"  - System: {system_mem_per_bl / (1024 * 1024):.2f} MB")

    # Calculate how many baselines can fit in GPU memory
    gpu_bl_limit = max(1, int(gpu_usable_mem // gpu_mem_per_bl))

    # Calculate how many baselines can fit in system memory
    system_bl_limit = max(1, int(system_usable_mem // system_mem_per_bl))

    # Take the minimum of the two limits
    bl_batch_size = min(gpu_bl_limit, system_bl_limit)

    logger.info("Based on memory constraints:")
    logger.info(f"  - GPU limit: {gpu_bl_limit} baselines")
    logger.info(f"  - System memory limit: {system_bl_limit} baselines")
    logger.info(f"  - Selected batch size: {bl_batch_size} baselines")

    # # Add a fixed upper limit to be safe
    # max_batch_size = 20
    # if bl_batch_size > max_batch_size:
    #     print(f"Limiting batch size to {max_batch_size} for safety")
    #     bl_batch_size = max_batch_size

    return bl_batch_size


def normalize_bandpass_with_polynomial_fit(
    bandpass, polynomial_degree=5, deviation_threshold=5.0
):
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
        # Not enough valid channels for polynomial fitting - use median filter
        try:
            smooth_bandpass = signal.medfilt(bandpass, kernel_size=15)
            smooth_valid = True
        except Exception:
            smooth_bandpass = bandpass.copy()
    else:
        try:
            # Pre-flag extremely high values that could skew the fit
            # This helps with those large spikes at channel 0
            temp_valid = valid_channels.copy()
            if np.any(valid_channels):
                median_val = np.median(bandpass[valid_channels])
                mad = np.median(np.abs(bandpass[valid_channels] - median_val))
                extreme_threshold = (
                    median_val + 10 * mad * 1.4826
                )  # Higher threshold for pre-flagging
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
        except Exception:
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
                rfi_mask[
                    np.abs(residuals - median_resid) > deviation_threshold * robust_std
                ] = True

    return smooth_bandpass, rfi_mask, smooth_valid


def process_baselines_batch_gpu(
    baseline_data, field_id, corr_to_process, options, freq_axis
):
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
    # Get logger from options
    logger = options.get("logger")

    total_start_time = time.time()

    # Create a main CUDA stream if GPU available
    if GPU_AVAILABLE:
        main_stream = cuda.stream()
        cuda.synchronize()  # Synchronize before starting
    else:
        main_stream = None

    # Extract baselines to process
    baselines = list(baseline_data.keys())
    logger.info(
        f"Processing batch of {len(baselines)} baselines with optimized GPU transfers..."
    )

    # Results dictionary
    results = {}

    # Dictionary to store diagnostic data
    diagnostic_data = {} if options["diagnostic_plots"] else None

    # Get polynomial degree for fitting from options or use default
    poly_degree = options.get("poly_degree", 5)
    deviation_threshold = options.get("deviation_threshold", 3.0)

    # Pre-calculate thresholds for all baselines
    logger.info("Normalizing and pre-calculating thresholds for all baselines...")
    threshold_arrays = {}

    # Use multiple threads for bandpass processing
    max_workers = min(options.get("max_threads", 16), len(baselines))
    logger.info(f"Using {max_workers} threads for parallel bandpass processing")

    # Function to process a single baseline for bandpass normalization
    def process_baseline_bandpass(bl):
        bl_data = baseline_data[bl]

        # Dictionary to store results
        bl_results = {}

        # Process different correlation configurations
        if (
            len(corr_to_process) == 2
            and corr_to_process[0] == 0
            and corr_to_process[-1] == bl_data.DATA.shape[2] - 1
        ):
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

            # Calculate bandpass (median across time for each channel) - OPTIMIZED
            bandpass = calculate_bandpass_parallel(vis_amp, existing_flags_combined)

            # Apply bandpass normalization using polynomial fitting
            smooth_bandpass, rfi_channels, fit_valid = (
                normalize_bandpass_with_polynomial_fit(
                    bandpass, poly_degree, deviation_threshold
                )
            )

            # Log RFI detection results
            rfi_count = np.sum(rfi_channels)
            if options.get("verbose"):
                logger.debug(
                    f"Detected {rfi_count} RFI channels in baseline {bl[0]}-{bl[1]} (combined pols)"
                )

            # Apply bandpass normalization and flagging - OPTIMIZED
            normalized_amp = vis_amp.copy()
            normalized_flags = existing_flags_combined.copy()

            # Apply normalization in parallel using Numba
            apply_bandpass_normalization_parallel(
                normalized_amp, normalized_flags, smooth_bandpass, rfi_channels
            )

            # Calculate robust thresholds on the normalized data
            base_thresholds = calculate_robust_thresholds(
                normalized_amp, normalized_flags, options["sigma_factor"]
            )

            bl_results[bl] = {
                "base_thresholds": base_thresholds,
                "vis_amp": normalized_amp,
                "existing_flags": normalized_flags,
                "mode": "combined",
                "bandpass": bandpass,
                "smooth_bandpass": smooth_bandpass,
                "rfi_channels": rfi_channels,
            }
        else:
            # Process each correlation separately
            for corr_idx in corr_to_process:
                vis_data = bl_data.DATA[:, :, corr_idx]
                existing_flags = bl_data.FLAG[:, :, corr_idx]

                # Calculate amplitude
                vis_amp = np.abs(vis_data)
                vis_amp = np.ma.masked_array(vis_amp, mask=existing_flags).filled(0)

                # Calculate bandpass (median across time for each channel) - OPTIMIZED
                bandpass = calculate_bandpass_parallel(vis_amp, existing_flags)

                # Apply bandpass normalization using polynomial fitting
                smooth_bandpass, rfi_channels, fit_valid = (
                    normalize_bandpass_with_polynomial_fit(
                        bandpass, poly_degree, deviation_threshold
                    )
                )

                # Log RFI detection results
                rfi_count = np.sum(rfi_channels)
                if options.get("verbose"):
                    logger.debug(
                        f"Detected {rfi_count} RFI channels in baseline {bl[0]}-{bl[1]}, pol {corr_idx}"
                    )

                # Apply bandpass normalization and flagging - OPTIMIZED
                normalized_amp = vis_amp.copy()
                normalized_flags = existing_flags.copy()

                # Apply normalization in parallel using Numba
                apply_bandpass_normalization_parallel(
                    normalized_amp, normalized_flags, smooth_bandpass, rfi_channels
                )

                # Calculate robust thresholds on the normalized data
                base_thresholds = calculate_robust_thresholds(
                    normalized_amp, normalized_flags, options["sigma_factor"]
                )

                # Create unique key for this baseline-correlation
                baseline_corr_key = (bl[0], bl[1], corr_idx)

                bl_results[baseline_corr_key] = {
                    "base_thresholds": base_thresholds,
                    "vis_amp": normalized_amp,
                    "existing_flags": normalized_flags,
                    "mode": "single",
                    "corr_idx": corr_idx,
                    "bl": bl,
                    "bandpass": bandpass,
                    "smooth_bandpass": smooth_bandpass,
                    "rfi_channels": rfi_channels,
                }

        return bl_results

    # Process baselines in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all baselines for processing
        future_to_bl = {
            executor.submit(process_baseline_bandpass, bl): bl for bl in baselines
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_bl):
            bl = future_to_bl[future]
            try:
                # Get results for this baseline
                bl_results = future.result()
                # Add to threshold_arrays
                threshold_arrays.update(bl_results)
            except Exception as e:
                logger.error(f"Error processing baseline {bl}: {str(e)}")
                import traceback

                traceback.print_exc()

    # Process each baseline/correlation with dedicated CUDA streams
    # Use a pool of streams for parallel execution
    if GPU_AVAILABLE:
        num_streams = min(5, len(threshold_arrays))  # Limit number of streams
        streams = [cuda.stream() for _ in range(num_streams)]
    else:
        num_streams = 1
        streams = [None]

    # Distribute work among streams
    stream_idx = 0
    for key, threshold_data in threshold_arrays.items():
        # Get the current stream
        stream = streams[stream_idx]

        # Get process info
        if threshold_data["mode"] == "single":
            corr_idx = threshold_data["corr_idx"]
            bl = threshold_data["bl"]
            info_string = f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Pol {corr_idx}"
        else:
            bl = key
            info_string = f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Combined Pols"

        # Get data
        vis_amp = threshold_data["vis_amp"]
        existing_flags = threshold_data["existing_flags"]
        base_thresholds = threshold_data["base_thresholds"]

        # Run GPU flagger with its own stream - using NORMALIZED data and precalculated thresholds
        flags, _ = sumthreshold_gpu(
            vis_amp,
            existing_flags,
            info_string,
            combinations=options["combinations"],
            sigma_factor=None,  # Not used with precalculated thresholds
            rho=options["rho"],
            diagnostic_plots=False,
            stream=stream,
            precalculated_thresholds=base_thresholds,
            logger=logger,
        )

        # Combine with existing flags
        combined_flags = np.logical_or(existing_flags, flags)

        # Store diagnostic data if enabled
        if options["diagnostic_plots"]:
            if threshold_data["mode"] == "single":
                corr_idx = threshold_data["corr_idx"]
                bl = threshold_data["bl"]
                diag_key = (bl, corr_idx)
            else:
                diag_key = (bl, "combined")

            # Store diagnostic data
            diagnostic_data[diag_key] = {
                "vis_amp": vis_amp.copy(),
                "combined_flags": combined_flags.copy(),
                "existing_flags": existing_flags.copy(),
                "corr_idx": "combined"
                if threshold_data["mode"] == "combined"
                else corr_idx,
                "bl": bl,
                "field_id": field_id,
                "bandpass": threshold_data["bandpass"],
                "smooth_bandpass": threshold_data["smooth_bandpass"],
                "rfi_channels": threshold_data["rfi_channels"],
            }

        if threshold_data["mode"] == "combined":
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
                        results[bl][:, :, corr_idx], combined_flags
                    )  # NumPy will auto-broadcast here
                else:
                    # If somehow combined_flags is already 3D
                    results[bl][:, :, corr_idx] = np.logical_or(
                        results[bl][:, :, corr_idx], combined_flags
                    )
        else:
            # Single correlation mode
            corr_idx = threshold_data["corr_idx"]
            orig_bl = threshold_data["bl"]

            if orig_bl not in results:
                # Initialize full flag array
                bl_data = baseline_data[orig_bl]
                full_flags = bl_data.FLAG.copy()
                results[orig_bl] = full_flags

            # Update this correlation's flags
            results[orig_bl][:, :, corr_idx] = np.logical_or(
                results[orig_bl][:, :, corr_idx], combined_flags
            )

        # Move to next stream
        stream_idx = (stream_idx + 1) % num_streams

    # Synchronize all streams
    if GPU_AVAILABLE:
        for stream in streams:
            stream.synchronize()
        # Final synchronization
        cuda.synchronize()

    # Generate diagnostic plots after processing
    if options["diagnostic_plots"] and diagnostic_data:
        logger.info(f"Generating {len(diagnostic_data)} diagnostic plots...")
        output_dir = options.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)

        for (bl, corr_idx), data in diagnostic_data.items():
            generate_diagnostic_plot(
                data["vis_amp"],
                data["combined_flags"],
                data["existing_flags"],
                bl,
                data["field_id"],
                corr_idx,
                freq_axis,
                output_dir,
            )

    total_time = time.time() - total_start_time
    logger.info(f"Batch processing completed in {total_time:.2f} seconds")

    return results


def generate_diagnostic_plot(
    vis_amp,
    combined_flags,
    existing_flags,
    bl,
    field_id,
    corr_idx,
    freq_axis,
    output_dir,
):
    """Generate a diagnostic plot showing flagged and unflagged data"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Convert corr_idx to string representation
        corr_str = str(corr_idx) if isinstance(corr_idx, (int, float)) else corr_idx

        # Create filename
        filename = os.path.join(
            output_dir, f"pooh_field{field_id}_bl{bl[0]}-{bl[1]}_pol{corr_str}.png"
        )

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
        plt.imshow(
            vis_amp_plot,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(label="Amplitude", shrink=0.8)
        plt.title("Raw Dynamic Spectra")
        plt.xlabel("Frequency")
        plt.ylabel("Time")

        # Right panel: Flagged spectra with existing vs new flags
        plt.subplot(1, 2, 2)

        # Plot the original data first
        plt.imshow(
            vis_amp_plot,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        # Subsample existing flags
        existing_flags_plot = existing_flags[::time_skip, ::freq_skip]

        # Calculate new flags (combined - existing)
        new_flags_plot = combined_flags_plot & ~existing_flags_plot

        # Create flag overlay with different colors
        flag_img = np.zeros(vis_amp_plot.shape + (4,), dtype=np.float32)  # RGBA

        # Existing flags in RED
        flag_img[existing_flags_plot, 0] = 1.0  # R
        flag_img[existing_flags_plot, 1] = 0.0  # G
        flag_img[existing_flags_plot, 2] = 0.0  # B
        flag_img[existing_flags_plot, 3] = 0.8  # Alpha

        # New POOH flags in CYAN
        flag_img[new_flags_plot, 0] = 0.0  # R
        flag_img[new_flags_plot, 1] = 1.0  # G
        flag_img[new_flags_plot, 2] = 1.0  # B
        flag_img[new_flags_plot, 3] = 0.8  # Alpha

        # Overlay the flags
        plt.imshow(flag_img, aspect="auto", origin="lower", interpolation="nearest")

        plt.colorbar(label="Amplitude", shrink=0.8)
        plt.title("RFI Flags: Existing vs POOH Detected")
        plt.xlabel("Frequency")
        plt.ylabel("Time")

        # Add legend
        import matplotlib.patches as mpatches

        red_patch = mpatches.Patch(color="red", label="Existing Flags")
        cyan_patch = mpatches.Patch(color="cyan", label="POOH Detected")
        plt.legend(handles=[red_patch, cyan_patch], loc="upper right")

        # Calculate percentages
        percent_existing = 100 * np.sum(existing_flags_plot) / existing_flags_plot.size
        percent_new = 100 * np.sum(new_flags_plot) / new_flags_plot.size
        percent_total = 100 * np.sum(combined_flags_plot) / combined_flags_plot.size

        plt.suptitle(
            f"BL {bl[0]}-{bl[1]}, Pol {corr_str}, Field {field_id}\n"
            f"Existing: {percent_existing:.1f}% | POOH: {percent_new:.1f}% | Total: {percent_total:.1f}%"
        )

        plt.tight_layout()
        plt.savefig(filename, dpi=80, bbox_inches="tight")
        plt.close()

    except Exception:
        pass
    finally:
        plt.close("all")
        gc.collect()


def process_baseline_async(
    bl_data,
    bl,
    field_id,
    corr_to_process,
    options,
    freq_axis,
    total_flagged,
    total_visibilities,
    stream=None,
    chunk_info="",
    output_dir="outputs",
):
    """Process a single baseline or chunk asynchronously"""
    # Get logger from options
    logger = options.get("logger")

    # Initialize time axis
    time_axis = np.arange(bl_data.sizes["row"])

    # Initialize flags array if needed
    if options["apply_flags"]:
        # CRITICAL: Start with original flags to preserve flags in correlations not being processed
        combined_flags = bl_data.FLAG.copy()
    else:
        combined_flags = None

    # Track flag counts
    existing_flag_count = 0
    new_flag_count = 0

    # Store diagnostic data
    diagnostic_data = []

    # Create a stream if not provided and GPU is available
    if GPU_AVAILABLE:
        if stream is None:
            stream = cuda.stream()
        # Synchronize to ensure clean GPU state
        cuda.synchronize()

    if len(corr_to_process) == 1:
        # Process a single correlation
        corr_idx = corr_to_process[0]

        # Extract data (use views where possible to avoid copying)
        vis_data = bl_data.DATA[
            :, :, corr_idx
        ]  # View - will be copied by masked_array if needed
        existing_flags = bl_data.FLAG[:, :, corr_idx]  # View - read-only
        existing_flag_count += np.sum(existing_flags)

        # Calculate amplitude (this creates a new array, so no copy needed above)
        masked_data = np.ma.masked_array(vis_data, mask=existing_flags)
        vis_amp = np.abs(masked_data.filled(0))

        # Skip if all flagged
        if np.all(existing_flags):
            if options["verbose"]:
                logger.debug(
                    f"All data already flagged for baseline {bl} corr {corr_idx} {chunk_info}. Skipping."
                )
            if options["apply_flags"]:
                combined_flags[:, :, corr_idx] = existing_flags
            return combined_flags, existing_flag_count, 0

        # Run SumThreshold
        new_flags, _ = sumthreshold_gpu(
            vis_amp,
            existing_flags,
            f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Pol {corr_idx} {chunk_info}",
            combinations=options["combinations"],
            sigma_factor=options["sigma_factor"],
            rho=options["rho"],
            diagnostic_plots=False,
            stream=stream,
            logger=logger,
        )

        # Combine flags
        combined_flags_corr = np.logical_or(existing_flags, new_flags)

        # Count new flags
        new_flag_count += np.sum(combined_flags_corr) - np.sum(existing_flags)

        # Store diagnostic data
        if options["diagnostic_plots"]:
            diagnostic_data.append(
                {
                    "vis_amp": vis_amp.copy(),
                    "combined_flags": combined_flags_corr.copy(),
                    "existing_flags": existing_flags.copy(),
                    "corr_idx": corr_idx,
                }
            )

        # Update total visibilities
        total_visibilities += vis_amp.size

        # Update combined flags
        if options["apply_flags"]:
            combined_flags[:, :, corr_idx] = combined_flags_corr

    elif (
        len(corr_to_process) == 2
        and corr_to_process[0] == 0
        and corr_to_process[-1] == bl_data.DATA.shape[2] - 1
    ):
        # Process combined correlations (Stokes I approximation) - use views
        vis_data_0 = bl_data.DATA[:, :, corr_to_process[0]]  # View
        vis_data_1 = bl_data.DATA[:, :, corr_to_process[1]]  # View
        existing_flags_0 = bl_data.FLAG[:, :, corr_to_process[0]]  # View
        existing_flags_1 = bl_data.FLAG[:, :, corr_to_process[1]]  # View

        # Count existing flags
        existing_flag_count += np.sum(existing_flags_0) + np.sum(existing_flags_1)

        # Replace flagged values with zeros (creates new arrays)
        masked_data_0 = np.ma.masked_array(vis_data_0, mask=existing_flags_0).filled(0)
        masked_data_1 = np.ma.masked_array(vis_data_1, mask=existing_flags_1).filled(0)

        # Sum data
        vis_sum = masked_data_0 + masked_data_1
        vis_amp = np.abs(vis_sum)

        # Combine existing flags
        existing_flags_combined = np.logical_or(existing_flags_0, existing_flags_1)

        # Skip if all flagged
        if np.all(existing_flags_combined):
            if options["verbose"]:
                logger.debug(
                    f"All data already flagged for baseline {bl} combined corrs {chunk_info}. Skipping."
                )
            if options["apply_flags"]:
                for corr_idx in corr_to_process:
                    combined_flags[:, :, corr_idx] = bl_data.FLAG[:, :, corr_idx].copy()
            return combined_flags, existing_flag_count, 0

        # Run SumThreshold
        new_flags, _ = sumthreshold_gpu(
            vis_amp,
            existing_flags_combined,
            f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Combined Pols {chunk_info}",
            combinations=options["combinations"],
            sigma_factor=options["sigma_factor"],
            rho=options["rho"],
            diagnostic_plots=False,
            stream=stream,
            logger=logger,
        )

        # Combine flags
        combined_flags_corr = np.logical_or(existing_flags_combined, new_flags)

        # Count new flags
        new_flag_count += (
            np.sum(combined_flags_corr) - np.sum(existing_flags_combined)
        ) * len(corr_to_process)

        # Store diagnostic data
        if options["diagnostic_plots"]:
            diagnostic_data.append(
                {
                    "vis_amp": vis_amp.copy(),
                    "combined_flags": combined_flags_corr.copy(),
                    "existing_flags": existing_flags_combined.copy(),
                    "corr_idx": "combined",
                }
            )

        # Update visibilities
        total_visibilities += vis_amp.size * len(corr_to_process)

        # Store flags
        if options["apply_flags"]:
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
                if options["verbose"]:
                    logger.debug(
                        f"All data already flagged for baseline {bl} corr {corr_idx} {chunk_info}. Skipping."
                    )
                if options["apply_flags"]:
                    combined_flags[:, :, corr_idx] = existing_flags
                continue

            # Run SumThreshold
            new_flags, _ = sumthreshold_gpu(
                vis_amp,
                existing_flags,
                f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Pol {corr_idx} {chunk_info}",
                combinations=options["combinations"],
                sigma_factor=options["sigma_factor"],
                rho=options["rho"],
                diagnostic_plots=False,
                stream=stream,
                logger=logger,
            )

            # Combine flags
            combined_flags_corr = np.logical_or(existing_flags, new_flags)

            # Count new flags
            new_flag_count += np.sum(combined_flags_corr) - np.sum(existing_flags)

            # Store diagnostic data
            if options["diagnostic_plots"]:
                diagnostic_data.append(
                    {
                        "vis_amp": vis_amp.copy(),
                        "combined_flags": combined_flags_corr.copy(),
                        "existing_flags": existing_flags.copy(),
                        "corr_idx": corr_idx,
                    }
                )

            # Update visibilities
            total_visibilities += vis_amp.size

            # Store flags
            if options["apply_flags"]:
                combined_flags[:, :, corr_idx] = combined_flags_corr

    # Generate diagnostic plots
    if options["diagnostic_plots"] and diagnostic_data:
        os.makedirs(output_dir, exist_ok=True)
        for data in diagnostic_data:
            generate_diagnostic_plot(
                data["vis_amp"],
                data["combined_flags"],
                data["existing_flags"],
                bl,
                field_id,
                data["corr_idx"],
                freq_axis,
                output_dir,
            )

    # Ensure synchronization before returning
    if GPU_AVAILABLE:
        if stream:
            stream.synchronize()
        cuda.synchronize()

    return combined_flags, existing_flag_count, new_flag_count


def process_single_field(
    ms_file, field_id, options, freq_axis, gpu_usable_mem, system_usable_mem
):
    """
    Process a single field independently. Can be called via multiprocessing.
    Returns statistics dictionary.
    """
    # Get logger from options
    logger = options.get("logger")

    # Initialize field statistics
    total_flagged = 0
    total_new_flags = 0
    total_visibilities = 0
    baselines_processed = 0
    baselines_skipped = 0

    # Get chunk size from options (user-configurable)
    chunk_size = options.get("chunk_size", 200000)

    logger.info(f"\n*** Processing Field {field_id} ***")

    # Get unique baselines for this field
    logger.info(f"Getting unique baselines for field {field_id}...")
    baselines = set()

    # Read antenna pairs with configurable chunk size
    taql_where = f"FIELD_ID={field_id}"
    xds_list = xds_from_ms(
        ms_file,
        columns=("ANTENNA1", "ANTENNA2"),
        taql_where=taql_where,
        chunks={"row": chunk_size},
    )

    # Extract unique baselines
    for ds in xds_list:
        ant1, ant2 = dask.compute(ds.ANTENNA1.data, ds.ANTENNA2.data)
        for a1, a2 in zip(ant1, ant2):
            baselines.add((a1, a2))

    baselines = list(baselines)
    logger.info(f"Found {len(baselines)} unique baselines in field {field_id}")

    # Skip if no baselines
    if not baselines:
        logger.warning(f"No baselines found for field {field_id}. Skipping.")
        return {
            "total_flagged": total_flagged,
            "total_new_flags": total_new_flags,
            "total_visibilities": total_visibilities,
            "baselines_processed": baselines_processed,
            "baselines_skipped": baselines_skipped,
        }

    # Get a sample baseline to check data shape
    sample_bl = baselines[0]
    sample_taql = (
        f"FIELD_ID={field_id} AND ANTENNA1={sample_bl[0]} AND ANTENNA2={sample_bl[1]}"
    )

    try:
        sample_ds = xds_from_ms(ms_file, columns=("DATA",), taql_where=sample_taql)[0]

        if sample_ds.sizes["row"] == 0:
            logger.warning(f"No data found for sample baseline {sample_bl}. Skipping field.")
            return {
                "total_flagged": total_flagged,
                "total_new_flags": total_new_flags,
                "total_visibilities": total_visibilities,
                "baselines_processed": baselines_processed,
                "baselines_skipped": baselines_skipped,
            }

        data_shape = sample_ds.DATA.shape
    except Exception as e:
        logger.error(f"Error reading sample baseline: {str(e)}")
        try:
            # Try simpler read
            simple_ds = xds_from_ms(ms_file, columns=("DATA",))[0]
            data_shape = simple_ds.DATA.shape
        except Exception as e2:
            logger.error(f"Could not determine data shape: {str(e2)}")
            return {
                "total_flagged": total_flagged,
                "total_new_flags": total_new_flags,
                "total_visibilities": total_visibilities,
                "baselines_processed": baselines_processed,
                "baselines_skipped": baselines_skipped,
            }

    logger.info(f"Data shape: {data_shape}")

    # Get number of correlations
    n_corr = data_shape[2]
    logger.info(f"Number of correlations: {n_corr}")

    # Determine correlations to process
    corr_to_process = options["corr_to_process"]
    if corr_to_process is None:
        # Default to first and last for Stokes I
        if n_corr >= 2:
            corr_to_process = [0, n_corr - 1]
            logger.info(f"Processing correlations 0 and {n_corr - 1} (default)")
        else:
            corr_to_process = [0]
            logger.info("Processing correlation 0 (default)")
    else:
        logger.info(f"Processing correlations: {corr_to_process}")

    # Calculate ideal batch size
    time_samples = sample_ds.sizes["row"]
    freq_channels = data_shape[1]

    logger.info(
        f"Typical baseline has {time_samples} time samples × {freq_channels} frequency channels"
    )

    # PRE-FILTER: Check which baselines are completely flagged (load only FLAG, not DATA)
    logger.info(f"\n[PRE-FILTER] Checking flag status for {len(baselines)} baselines...")
    valid_baselines_set = set()  # Use set to avoid duplicates
    baseline_flag_map = {}  # Map baseline -> is_completely_flagged

    # Build query for all baselines at once
    baseline_clauses = []
    for ant1, ant2 in baselines:
        baseline_clauses.append(f"(ANTENNA1={ant1} AND ANTENNA2={ant2})")

    taql_where_all = f"FIELD_ID={field_id} AND ({' OR '.join(baseline_clauses)})"

    try:
        # Read only FLAG column for all baselines with configurable chunks
        flag_ds_list = xds_from_ms(
            ms_file,
            columns=("FLAG", "ANTENNA1", "ANTENNA2"),
            taql_where=taql_where_all,
            chunks={"row": chunk_size},
        )

        # Check each baseline
        for ds in flag_ds_list:
            ant1, ant2, flags = dask.compute(
                ds.ANTENNA1.data, ds.ANTENNA2.data, ds.FLAG.data
            )

            # Group by baseline
            baseline_flags = {}
            for i, (a1, a2) in enumerate(zip(ant1, ant2)):
                bl = (a1, a2)
                if bl not in baseline_flags:
                    baseline_flags[bl] = []
                baseline_flags[bl].append(flags[i])

            # Check if completely flagged
            for bl, flag_list in baseline_flags.items():
                combined_flags = np.concatenate(flag_list, axis=0)
                is_completely_flagged = np.all(combined_flags)
                baseline_flag_map[bl] = is_completely_flagged

                if is_completely_flagged:
                    if options["verbose"]:
                        logger.debug(f"  Baseline {bl}: 100% flagged - SKIP")
                    baselines_skipped += 1
                else:
                    if options["verbose"]:
                        percent_flagged = (
                            100 * np.sum(combined_flags) / combined_flags.size
                        )
                        logger.debug(
                            f"  Baseline {bl}: {percent_flagged:.1f}% flagged - PROCESS"
                        )
                    valid_baselines_set.add(bl)  # Use set to avoid duplicates

        valid_baselines = list(valid_baselines_set)  # Convert to list
        logger.info(
            f"[PRE-FILTER] Result: {len(valid_baselines)} valid, {baselines_skipped} completely flagged"
        )

    except Exception as e:
        logger.warning(
            f"[PRE-FILTER] Warning: Pre-filter failed ({str(e)}), processing all baselines"
        )
        valid_baselines = baselines  # Fall back to processing all

    # Skip if no valid baselines
    if not valid_baselines:
        logger.warning(f"No valid baselines to process in field {field_id}.")
        return {
            "total_flagged": total_flagged,
            "total_new_flags": total_new_flags,
            "total_visibilities": total_visibilities,
            "baselines_processed": baselines_processed,
            "baselines_skipped": baselines_skipped,
        }

    # Process baselines in batches where possible
    baseline_data = {}  # Dictionary to store baseline data

    # Try batch processing first
    try:
        # Calculate batch size
        baseline_data[sample_bl] = sample_ds  # Add sample for calculation
        bl_per_batch = calculate_baseline_batch_size(
            baseline_data, sample_bl, options, gpu_usable_mem, system_usable_mem
        )

        if bl_per_batch > 1:
            # Batch processing is possible
            for i in range(0, len(valid_baselines), bl_per_batch):
                batch = valid_baselines[i : i + bl_per_batch]
                batch_size = len(batch)

                logger.info(
                    f"\nProcessing batch {i // bl_per_batch + 1}/{(len(valid_baselines) + bl_per_batch - 1) // bl_per_batch}: {batch_size} baselines"
                )

                # Read all baselines in one query
                baseline_clauses = []
                for ant1, ant2 in batch:
                    baseline_clauses.append(f"(ANTENNA1={ant1} AND ANTENNA2={ant2})")

                taql_where = (
                    f"FIELD_ID={field_id} AND ({' OR '.join(baseline_clauses)})"
                )

                # Read data
                batch_ds_list = xds_from_ms(
                    ms_file,
                    columns=("DATA", "FLAG", "ANTENNA1", "ANTENNA2"),
                    taql_where=taql_where,
                )

                # Dictionary to store baseline data
                baseline_data = {}

                # Process each dataset
                for ds in batch_ds_list:
                    # Materialize data in one operation
                    ant1, ant2, data, flags = dask.compute(
                        ds.ANTENNA1.data, ds.ANTENNA2.data, ds.DATA.data, ds.FLAG.data
                    )

                    # Group by baseline
                    for i, (a1, a2) in enumerate(zip(ant1, ant2)):
                        bl = (a1, a2)

                        if bl not in batch:
                            continue

                        if bl not in baseline_data:
                            baseline_data[bl] = {"data": [], "flags": [], "indices": []}

                        baseline_data[bl]["data"].append(data[i])
                        baseline_data[bl]["flags"].append(flags[i])
                        baseline_data[bl]["indices"].append(i)

                # Create data objects
                valid_baseline_data = {}

                for bl in batch:
                    if bl not in baseline_data or not baseline_data[bl]["data"]:
                        if options["verbose"]:
                            logger.debug(f"No data found for baseline {bl}. Skipping.")
                        baselines_skipped += 1
                        continue

                    # Combine data
                    bl_data = np.array(baseline_data[bl]["data"])
                    bl_flags = np.array(baseline_data[bl]["flags"])

                    # Note: Completely flagged baselines already filtered out in pre-filter step

                    # Create data object
                    class MaterializedData:
                        def __init__(self, data, flags, row_size):
                            self.DATA = data
                            self.FLAG = flags
                            self.sizes = {"row": row_size}

                    materialized_bl_data = MaterializedData(
                        bl_data, bl_flags, len(bl_data)
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
                        freq_axis,
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
                            logger.error(
                                f"Error processing results for baseline {bl}: {str(e)}"
                            )
                            if options["verbose"]:
                                import traceback

                                traceback.print_exc()
                            baselines_skipped += 1
                            continue

                # Write flags if requested
                if options["apply_flags"]:
                    try:
                        # Process each baseline separately
                        for bl in batch_results.keys():
                            if options["verbose"]:
                                logger.debug(f"Writing flags for baseline {bl}...")

                            # Create baseline-specific query
                            bl_taql = f"FIELD_ID={field_id} AND ANTENNA1={bl[0]} AND ANTENNA2={bl[1]}"

                            # Read original data
                            orig_ds_list = xds_from_ms(
                                ms_file,
                                columns=("FLAG",),
                                taql_where=bl_taql,
                            )

                            if not orig_ds_list or orig_ds_list[0].sizes["row"] == 0:
                                logger.warning(f"No data found for baseline {bl}. Skipping.")
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
                                        orig_flags[:, :, corr_idx], bl_flags
                                    )
                            else:
                                # If we already have 3D flags from processing specific correlations
                                # Make sure all correlations get the flags
                                if (
                                    len(bl_flags.shape) == 3
                                    and bl_flags.shape[2] < orig_flags.shape[2]
                                ):
                                    combined_flags = orig_flags.copy()

                                    # Apply processed correlation flags to all correlations
                                    processed_flags = np.any(
                                        bl_flags, axis=2, keepdims=True
                                    )
                                    for corr_idx in range(orig_flags.shape[2]):
                                        combined_flags[:, :, corr_idx] = np.logical_or(
                                            orig_flags[:, :, corr_idx],
                                            processed_flags[:, :, 0],
                                        )
                                else:
                                    # If shapes already match
                                    combined_flags = np.logical_or(orig_flags, bl_flags)

                            # Convert back to dask array with SAME chunking
                            new_flags_dask = da.from_array(
                                combined_flags, chunks=orig_ds.FLAG.data.chunks
                            )

                            # Create updated dataset
                            updated_ds = orig_ds.assign(
                                FLAG=(orig_ds.FLAG.dims, new_flags_dask)
                            )

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
                        logger.error(f"Error writing flags: {str(e)}")
                        import traceback

                        traceback.print_exc()
                else:
                    logger.warning("No valid baselines found in this batch. Skipping.")

                # Clean up
                batch_ds_list = None
                baseline_data = None
                valid_baseline_data = None
                batch_results = None

                # Force gc
                gc.collect()

                # Ensure GPU is cleared
                if GPU_AVAILABLE:
                    cuda.synchronize()
        else:
            # Single baseline processing
            raise ValueError("Forcing single baseline processing")
    except Exception as e:
        # Fall back to single baseline processing
        logger.info(
            f"Batch processing failed: {str(e)}. Falling back to single baseline processing."
        )

        # Process each baseline
        for bl_idx, bl in enumerate(tqdm(valid_baselines, desc="Baselines")):
            if options["verbose"]:
                logger.debug(
                    f"\nProcessing baseline {bl_idx + 1}/{len(valid_baselines)}: Antennas {bl[0]}-{bl[1]}"
                )

            try:
                # Read baseline data
                taql_where = (
                    f"FIELD_ID={field_id} AND ANTENNA1={bl[0]} AND ANTENNA2={bl[1]}"
                )

                bl_ds_list = xds_from_ms(
                    ms_file, columns=("DATA", "FLAG"), taql_where=taql_where
                )

                # Skip if no data
                if not bl_ds_list or bl_ds_list[0].sizes["row"] == 0:
                    if options["verbose"]:
                        logger.debug(f"No data found for baseline {bl}. Skipping.")
                    baselines_skipped += 1
                    continue

                # Get data
                bl_data, bl_flags = dask.compute(
                    bl_ds_list[0].DATA.data, bl_ds_list[0].FLAG.data
                )

                # Note: Completely flagged baselines already filtered out in pre-filter step

                # Create data object
                class MaterializedData:
                    def __init__(self, data, flags, row_size):
                        self.DATA = data
                        self.FLAG = flags
                        self.sizes = {"row": row_size}

                materialized_bl_data = MaterializedData(
                    bl_data, bl_flags, bl_data.shape[0]
                )

                # Process this baseline
                # Create a dedicated stream (if GPU available)
                stream = cuda.stream() if GPU_AVAILABLE else None

                # Process baseline
                all_flags, baseline_existing, baseline_new = process_baseline_async(
                    materialized_bl_data,
                    bl,
                    field_id,
                    corr_to_process,
                    options,
                    freq_axis,
                    total_flagged,
                    total_visibilities,
                    stream,
                    output_dir=options.get("output_dir", "outputs"),
                )

                # Update statistics
                total_flagged += baseline_existing
                total_new_flags += baseline_new
                total_visibilities += np.prod(bl_data.shape)
                baselines_processed += 1

                # Write flags if requested
                if options["apply_flags"] and all_flags is not None:
                    try:
                        # Read original data
                        orig_ds_list = xds_from_ms(
                            ms_file, columns=("FLAG",), taql_where=taql_where
                        )

                        if orig_ds_list:
                            # Get original flags
                            orig_ds = orig_ds_list[0]
                            orig_flags = orig_ds.FLAG

                            # Combine flags
                            orig_flags_array = orig_ds.FLAG.data.compute()
                            if len(orig_flags_array.shape) != len(all_flags.shape):
                                if options["verbose"]:
                                    logger.debug(
                                        f"Handling dimension mismatch. Original: {orig_flags_array.shape}, New: {all_flags.shape}"
                                    )

                                # Create a properly sized output array
                                combined_flags = np.copy(orig_flags_array)

                                if (
                                    len(orig_flags_array.shape) == 3
                                    and len(all_flags.shape) == 2
                                ):
                                    # If original is 3D and new is 2D
                                    for corr_idx in range(orig_flags_array.shape[2]):
                                        # Apply 2D flags to each correlation plane
                                        combined_flags[:, :, corr_idx] = np.logical_or(
                                            orig_flags_array[:, :, corr_idx], all_flags
                                        )
                                else:
                                    logger.warning(
                                        "Unexpected dimension mismatch. Using original flags."
                                    )
                                    combined_flags = orig_flags_array
                            else:
                                # Normal case - same dimensions
                                combined_flags = np.logical_or(
                                    orig_flags_array, all_flags
                                )

                            # Convert back to dask array
                            new_flags_dask = da.from_array(
                                combined_flags, chunks=orig_flags.data.chunks
                            )

                            # Create updated dataset
                            updated_ds = orig_ds.assign(
                                FLAG=(orig_flags.dims, new_flags_dask)
                            )

                            # Write back
                            if options["verbose"]:
                                logger.debug(f"Writing flags for baseline {bl}...")
                            write_back = xds_to_table([updated_ds], ms_file, ["FLAG"])
                            dask.compute(write_back)
                    except Exception as e:
                        logger.error(f"Error writing flags for baseline {bl}: {str(e)}")
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
                if GPU_AVAILABLE:
                    cuda.synchronize()

            except Exception as e:
                logger.error(f"Error processing baseline {bl}: {str(e)}")
                if options["verbose"]:
                    import traceback

                    traceback.print_exc()
                baselines_skipped += 1
                continue

    # Return field statistics
    return {
        "total_flagged": total_flagged,
        "total_new_flags": total_new_flags,
        "total_visibilities": total_visibilities,
        "baselines_processed": baselines_processed,
        "baselines_skipped": baselines_skipped,
    }


def hunt_ms(ms_file, options):
    """The main function which processes the MS file with SumThreshold flagging"""
    # Get logger from options
    logger = options.get("logger")

    # Clean up stale lock file from previous run (only in MS root, not subdirectories)
    lock_file = os.path.join(ms_file, "table.lock")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            logger.info("Removed stale lock file from MS directory")
        except Exception as e:
            logger.warning(f"Could not remove lock file: {e}")

    # Create output directory if needed
    if options["diagnostic_plots"]:
        output_dir = options.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)

    # Get antenna info
    antenna_ds = xds_from_table(f"{ms_file}::ANTENNA")[0]
    logger.info("\nANTENNA TABLE INFO")
    logger.info(f"   Number of antennas: {antenna_ds.sizes['row']}")
    if options.get("verbose"):
        logger.debug(f"   Antenna names: {antenna_ds.NAME.values}")

    # Get spectral window info
    spw_ds = xds_from_table(f"{ms_file}::SPECTRAL_WINDOW")[0]
    logger.info("\nSPECTRAL WINDOW INFO")
    logger.info(f"   Number of channels: {spw_ds.NUM_CHAN.values[0]}")
    logger.info(f"   Channel width: {spw_ds.CHAN_WIDTH.values[0][0]:.2f} Hz")
    if options.get("verbose"):
        logger.debug(
            f"   Channel frequencies: {spw_ds.CHAN_FREQ.values[0][:5]}... (first 5)"
        )

    # Get frequency axis
    freq_axis = spw_ds.CHAN_FREQ.values[0]

    # Get memory information
    logger.info("\nMEMORY ALLOCATION")
    gpu_usable_mem, system_usable_mem = get_memory_info(logger)

    # Track statistics
    total_start_time = time.time()
    total_flagged = 0
    total_new_flags = 0
    total_visibilities = 0
    baselines_processed = 0
    baselines_skipped = 0

    # Get field IDs
    logger.info("\nDiscovering fields...")
    try:
        field_ds = xds_from_table(f"{ms_file}::FIELD")[0]
        field_ids = np.arange(field_ds.sizes["row"])
        logger.info(f"   Found {len(field_ids)} field(s): {field_ids}")
    except Exception as e:
        logger.warning(f"Could not extract field IDs from MS: {str(e)}")
        logger.info("   Defaulting to field 0")
        field_ids = [0]  # Default to field 0

    # Determine if we can process fields in parallel (informational)
    # DISABLED: Field-level parallelization causes MS locking issues
    # when multiple processes try to read from the same MS file
    parallel_fields = 1  # Force sequential field processing

    # Keep this for future reference when we implement proper pre-loading
    # parallel_fields = determine_parallel_field_count(
    #     ms_file, field_ids, system_usable_mem, logger=logger
    # )

    # FIELD-LEVEL PARALLELIZATION
    if parallel_fields > 1 and len(field_ids) > 1:
        logger.info(f"\n[PARALLEL MODE] Processing {parallel_fields} fields simultaneously")
        logger.info(f"[PARALLEL MODE] Worker processes: {parallel_fields}\n")

        from multiprocessing import Pool

        # Process in batches
        for batch_start in range(0, len(field_ids), parallel_fields):
            batch_ids = field_ids[batch_start : batch_start + parallel_fields]
            logger.info(f"\n{'=' * 70}")
            logger.info(
                f"Batch {batch_start // parallel_fields + 1}: Processing fields {batch_ids}"
            )
            logger.info(f"{'=' * 70}\n")

            args_list = [
                (ms_file, fid, options, freq_axis, gpu_usable_mem, system_usable_mem)
                for fid in batch_ids
            ]

            with Pool(processes=len(batch_ids)) as pool:
                batch_results = pool.starmap(process_single_field, args_list)

            # Aggregate
            for stats in batch_results:
                total_flagged += stats["total_flagged"]
                total_new_flags += stats["total_new_flags"]
                total_visibilities += stats["total_visibilities"]
                baselines_processed += stats["baselines_processed"]
                baselines_skipped += stats["baselines_skipped"]
    else:
        logger.info(f"\n[SEQUENTIAL MODE] Processing {len(field_ids)} field(s)\n")
        for field_id in field_ids:
            stats = process_single_field(
                ms_file, field_id, options, freq_axis, gpu_usable_mem, system_usable_mem
            )
            total_flagged += stats["total_flagged"]
            total_new_flags += stats["total_new_flags"]
            total_visibilities += stats["total_visibilities"]
            baselines_processed += stats["baselines_processed"]
            baselines_skipped += stats["baselines_skipped"]

    # Calculate final statistics
    total_time = time.time() - total_start_time
    overall_percent_flagged = (
        100 * (total_flagged + total_new_flags) / total_visibilities
        if total_visibilities > 0
        else 0
    )
    new_percent_flagged = (
        100 * total_new_flags / total_visibilities if total_visibilities > 0 else 0
    )

    # Force garbage collection to ensure tables are closed
    import gc
    gc.collect()

    # Return results
    results = {
        "total_processing_time": total_time,
        "overall_percent_flagged": overall_percent_flagged,
        "new_percent_flagged": new_percent_flagged,
        "total_visibilities": total_visibilities,
        "existing_flags": total_flagged,
        "new_flags": total_new_flags,
        "baselines_processed": baselines_processed,
        "baselines_skipped": baselines_skipped,
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
