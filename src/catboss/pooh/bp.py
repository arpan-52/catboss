import gc
import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import time
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

# Check GPU availability and import Numba
try:
    from numba import cuda, jit

    GPU_AVAILABLE = cuda.is_available()
    if GPU_AVAILABLE:
        print("GPU detected - will use GPU acceleration")
    else:
        print("No GPU detected - will use CPU processing")
except Exception as e:
    print(f"GPU check failed: {e} - will use CPU processing")
    from numba import jit

    GPU_AVAILABLE = False
    cuda = None


def read_data_column(ms_file, datacolumn, columns_tuple, taql_where=None, chunks=None):
    """
    Read data from MS with support for RESIDUAL_DATA calculation.

    Args:
        ms_file: Path to MS file
        datacolumn: Column name to read (e.g., 'DATA', 'CORRECTED_DATA', 'RESIDUAL_DATA')
        columns_tuple: Tuple of columns to read (e.g., ('DATA', 'FLAG', 'ANTENNA1', 'ANTENNA2'))
        taql_where: Optional TAQL query string
        chunks: Optional chunking specification

    Returns:
        List of xarray datasets with data column appropriately set
    """
    # Special handling for RESIDUAL_DATA
    if datacolumn.upper() == 'RESIDUAL_DATA':
        # Read both DATA and MODEL_DATA
        cols_list = list(columns_tuple)
        # Replace 'DATA' with both 'DATA' and 'MODEL_DATA'
        if 'DATA' in cols_list:
            cols_list.remove('DATA')
        cols_with_model = tuple(cols_list + ['DATA', 'MODEL_DATA'])

        kwargs = {'columns': cols_with_model}
        if taql_where:
            kwargs['taql_where'] = taql_where
        if chunks:
            kwargs['chunks'] = chunks

        ds_list = xds_from_ms(ms_file, **kwargs)

        # Calculate residuals for each dataset
        result_ds_list = []
        for ds in ds_list:
            # Check if MODEL_DATA exists
            if not hasattr(ds, 'MODEL_DATA'):
                raise ValueError(
                    f"RESIDUAL_DATA requested but MODEL_DATA column not found in MS file. "
                    f"Please run a calibration task first or use a different data column."
                )

            # Calculate DATA - MODEL_DATA
            residual_data = ds.DATA - ds.MODEL_DATA

            # Create new dataset with DATA replaced by residuals
            ds_modified = ds.assign(DATA=residual_data)
            # Drop MODEL_DATA to save memory
            ds_modified = ds_modified.drop_vars('MODEL_DATA')
            result_ds_list.append(ds_modified)

        return result_ds_list

    else:
        # Normal case - just read the specified column
        # Replace 'DATA' in columns_tuple with the user-specified column
        cols_list = list(columns_tuple)
        if 'DATA' in cols_list:
            cols_list[cols_list.index('DATA')] = datacolumn
        cols_final = tuple(cols_list)

        kwargs = {'columns': cols_final}
        if taql_where:
            kwargs['taql_where'] = taql_where
        if chunks:
            kwargs['chunks'] = chunks

        try:
            ds_list = xds_from_ms(ms_file, **kwargs)

            # Rename the column back to 'DATA' for consistent processing
            result_ds_list = []
            for ds in ds_list:
                if hasattr(ds, datacolumn) and datacolumn != 'DATA':
                    ds_modified = ds.rename({datacolumn: 'DATA'})
                    result_ds_list.append(ds_modified)
                else:
                    result_ds_list.append(ds)

            return result_ds_list

        except Exception as e:
            raise ValueError(
                f"Failed to read column '{datacolumn}' from MS file. "
                f"Column may not exist. Error: {str(e)}"
            )


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


# ==================== GPU IMPLEMENTATIONS ====================
# @cuda.jit
# def sum_threshold_kernel_time_channel(amp, flags, thresholds, M):
#     """CUDA kernel for SumThreshold in time direction with channel-specific thresholds"""
#     # Get thread position
#     i, j = cuda.grid(2)

#     # Check if within bounds (including possible combinations)
#     if i < amp.shape[0] and j < amp.shape[1] - M + 1:
#         # Get threshold for this channel
#         threshold = thresholds[j]

#         # Get the current group
#         group_sum = 0.0

#         # Sum the group, replacing flagged values with threshold
#         for k in range(M):
#             if flags[i, j+k]:
#                 group_sum += thresholds[j+k]  # Use the channel's threshold
#             else:
#                 group_sum += amp[i, j+k]

#         # Calculate average threshold for this group
#         avg_threshold = 0.0
#         for k in range(M):
#             avg_threshold += thresholds[j+k]
#         avg_threshold /= M

#         # Check if the sum exceeds the threshold
#         if group_sum > M * avg_threshold:
#             # Flag samples that exceed the threshold
#             for k in range(M):
#                 if not flags[i, j+k] and amp[i, j+k] > thresholds[j+k]:
#                     flags[i, j+k] = True

# @cuda.jit
# def sum_threshold_kernel_freq_channel(amp, flags, thresholds, M):
#     """CUDA kernel for SumThreshold in frequency direction with channel-specific thresholds"""
#     # Get thread position
#     i, j = cuda.grid(2)

#     # Check if within bounds (including possible combinations)
#     if i < amp.shape[0] - M + 1 and j < amp.shape[1]:
#         # Get threshold for this channel
#         threshold = thresholds[j]

#         # Get the current group
#         group_sum = 0.0

#         # Sum the group, replacing flagged values with threshold
#         for k in range(M):
#             if flags[i+k, j]:
#                 group_sum += threshold  # All in same channel, use same threshold
#             else:
#                 group_sum += amp[i+k, j]

#         # Check if the sum exceeds the threshold
#         if group_sum > M * threshold:
#             # Flag samples that exceed the threshold
#             for k in range(M):
#                 if not flags[i+k, j] and amp[i+k, j] > threshold:
#                     flags[i+k, j] = True

if GPU_AVAILABLE:

    @cuda.jit
    def sum_threshold_kernel_time_channel(amp, flags, thresholds, M):
        i, j = cuda.grid(2)

        if i < amp.shape[0] and j < amp.shape[1] - M + 1:
            # Get threshold for this channel
            threshold = thresholds[j]

            # Get the current group
            group_sum = 0.0
            count = 0  # Count of unflagged points

            # Sum only unflagged values in the group
            for k in range(M):
                if not flags[i, j + k]:
                    group_sum += amp[i, j + k]
                    count += 1

            # Calculate average threshold for this group
            avg_threshold = 0.0
            for k in range(M):
                avg_threshold += thresholds[j + k]
            avg_threshold /= M

            # Only proceed if we have unflagged points
            if count > 0:
                # Check if the average of unflagged values exceeds the threshold
                if (group_sum / count) > avg_threshold:
                    # Flag samples that exceed the threshold
                    for k in range(M):
                        if not flags[i, j + k] and amp[i, j + k] > thresholds[j + k]:
                            flags[i, j + k] = True

    @cuda.jit
    def sum_threshold_kernel_freq_channel(amp, flags, thresholds, M):
        """CUDA kernel for SumThreshold in frequency direction with channel-specific thresholds"""
        # Get thread position
        i, j = cuda.grid(2)

        # Check if within bounds (including possible combinations)
        if i < amp.shape[0] - M + 1 and j < amp.shape[1]:
            # Get threshold for this channel
            threshold = thresholds[j]

            # Get the current group
            group_sum = 0.0
            count = 0  # Count of unflagged points

            # Sum only unflagged values in the group
            for k in range(M):
                if not flags[i + k, j]:
                    group_sum += amp[i + k, j]
                    count += 1

            # Only proceed if we have unflagged points
            if count > 0:
                # Check if the average of unflagged values exceeds the threshold
                if (group_sum / count) > threshold:
                    # Flag samples that exceed the threshold
                    for k in range(M):
                        if not flags[i + k, j] and amp[i + k, j] > threshold:
                            flags[i + k, j] = True


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
):
    """
    SumThreshold implementation with CPU/GPU support.
    Now supports respecting existing flags by treating flagged data points specially
    in the summing process.

    Args:
        amp: Amplitude array (2D: time × frequency)
        existing_flags: Boolean array of existing flags (same shape as amp)
        baseline_info: String description of the baseline for logging
        combinations: List of window sizes to use for SumThreshold
        sigma_factor: Multiplier for channel-based standard deviation in threshold calculation
        rho: Factor to reduce threshold for larger window sizes
        diagnostic_plots: Whether to generate diagnostic plots
        stream: CUDA stream to use for asynchronous execution (GPU only)
        precalculated_thresholds: Optional pre-calculated threshold array

    Returns:
        tuple: (flags, baseline_info) where flags is a boolean array of the same shape as amp
    """

    # Handle stream only if GPU is available
    if GPU_AVAILABLE:
        if stream is None:
            stream = cuda.stream()

    print(f"Starting the scratches on the {baseline_info}")

    # Set default combinations if not provided
    if combinations is None:
        combinations = [1, 2, 4, 8, 16]

    if rho is None:
        rho = 1.5

    print(f"Using the combinations: {combinations}")
    print(f"Using the rho value: {rho}")

    # Use pre-calculated thresholds if provided, otherwise calculate them
    if precalculated_thresholds is not None:
        channel_thresholds = precalculated_thresholds
        print("Using pre-calculated channel thresholds")
    else:
        print(
            "Calculating channel-based statistics on CPU (respecting existing flags)..."
        )
        channel_medians = np.zeros(amp.shape[1], dtype=np.float32)
        channel_stds = np.zeros(amp.shape[1], dtype=np.float32)

        # Calculate statistics while respecting existing flags
        for j in range(amp.shape[1]):
            # Get data for this channel, masking existing flags
            channel_data = amp[:, j]
            mask = existing_flags[:, j]
            unflagged_data = channel_data[~mask]

            if len(unflagged_data) > 0:
                # Calculate statistics only on unflagged data
                channel_medians[j] = np.median(unflagged_data)
                channel_stds[j] = np.std(unflagged_data)
            else:
                # If all data is flagged in this channel, use reasonable defaults
                # Use median and std of entire dataset or some default values
                all_unflagged = amp[~existing_flags]
                if len(all_unflagged) > 0:
                    channel_medians[j] = np.median(all_unflagged)
                    channel_stds[j] = np.std(all_unflagged)
                else:
                    # If everything is flagged, use reasonable defaults
                    channel_medians[j] = 0.0
                    channel_stds[j] = 1.0

        channel_thresholds = channel_medians + sigma_factor * channel_stds

    print(
        f"Channel thresholds - min: {np.min(channel_thresholds):.4f}, max: {np.max(channel_thresholds):.4f}"
    )

    # Initialize flags with existing flags
    flags = existing_flags.copy()

    # Process each combination size
    start_time = time.time()

    if GPU_AVAILABLE:
        # GPU path
        # Copy data and flags to GPU with specified stream
        d_data = cuda.to_device(amp, stream=stream)
        d_flags = cuda.to_device(flags, stream=stream)

        # Define thread block and grid dimensions
        threads_per_block = (16, 16)
        blocks_per_grid_x = (
            amp.shape[0] + threads_per_block[0] - 1
        ) // threads_per_block[0]
        blocks_per_grid_y = (
            amp.shape[1] + threads_per_block[1] - 1
        ) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        for M in combinations:
            # Calculate thresholds for this combination size
            combo_channel_thresholds = channel_thresholds / (rho ** np.log2(M))

            print(
                f"Processing combination size {M}, avg threshold: {np.mean(combo_channel_thresholds):.4f}"
            )

            # Copy thresholds to GPU using the specified stream
            d_thresholds = cuda.to_device(combo_channel_thresholds, stream=stream)

            # Time direction
            sum_threshold_kernel_time_channel[
                blocks_per_grid, threads_per_block, stream
            ](d_data, d_flags, d_thresholds, M)

            # Frequency direction
            sum_threshold_kernel_freq_channel[
                blocks_per_grid, threads_per_block, stream
            ](d_data, d_flags, d_thresholds, M)

        # Copy results back to host using the specified stream
        d_flags.copy_to_host(flags, stream=stream)
    else:
        # CPU path
        for M in combinations:
            # Calculate thresholds for this combination size
            combo_channel_thresholds = channel_thresholds / (rho ** np.log2(M))

            print(
                f"Processing combination size {M}, avg threshold: {np.mean(combo_channel_thresholds):.4f}"
            )

            # Time direction processing
            sum_threshold_cpu_time_channel(amp, flags, combo_channel_thresholds, M)

            # Frequency direction processing
            sum_threshold_cpu_freq_channel(amp, flags, combo_channel_thresholds, M)

    # Process results and generate statistics
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.4f} seconds")

    # Calculate percentage of newly flagged data (excluding existing flags)
    existing_count = np.sum(existing_flags)
    total_count = flags.size
    new_flags_count = np.sum(flags) - existing_count
    percent_new_flagged = (
        100 * new_flags_count / (total_count - existing_count)
        if (total_count - existing_count) > 0
        else 0
    )

    print(
        f"Found {new_flags_count} new flags ({percent_new_flagged:.2f}% of unflagged data) for the baseline {baseline_info}"
    )

    return flags, baseline_info


def get_memory_info():
    """Get information about available GPU and system memory"""
    # Get GPU memory information
    try:
        gpu_device = cuda.get_current_device()

        # Try different ways to get memory info
        try:
            gpu_mem = gpu_device.total_memory
        except AttributeError:
            try:
                # Alternative approach
                gpu_mem = cuda.current_context().get_memory_info()[1]  # total memory
            except:
                # Another approach sometimes used
                mem_info = cuda.cuda.cudart().cudaMemGetInfo()
                gpu_mem = mem_info[1]  # total memory

        gpu_usable_mem = gpu_mem * 0.8  # Use 80% of GPU memory to be safe
        print(
            f"GPU memory: {gpu_mem / 1e9:.2f} GB (usable: {gpu_usable_mem / 1e9:.2f} GB)"
        )
    except Exception as e:
        print(f"Warning: Could not determine GPU memory, assuming 8GB: {str(e)}")
        gpu_usable_mem = 6 * 1024 * 1024 * 1024 * 0.8  # Assume 8GB, use 80%

    # Get system memory information
    try:
        system_mem = psutil.virtual_memory()
        total_system_mem = system_mem.total
        available_system_mem = system_mem.available
        system_usable_mem = available_system_mem * 0.8  # Use 80% of available memory
        print(
            f"System memory: {total_system_mem / 1e9:.2f} GB (available: {available_system_mem / 1e9:.2f} GB, usable: {system_usable_mem / 1e9:.2f} GB)"
        )
    except Exception as e:
        print(f"Warning: Could not determine system memory, assuming 16GB: {str(e)}")
        system_usable_mem = 16 * 1024 * 1024 * 1024 * 0.8  # Assume 16GB, use 80%

    return gpu_usable_mem, system_usable_mem


def calculate_baseline_batch_size(
    baseline_data, sample_bl, options, gpu_usable_mem, system_usable_mem
):
    """Calculate how many baselines we can process at once based on memory constraints"""
    # Get a sample baseline to estimate memory requirements
    if sample_bl not in baseline_data:
        print("Sample baseline not found in data. Cannot estimate memory requirements.")
        return 1

    sample_data = baseline_data[sample_bl]

    # Get data dimensions
    time_samples = sample_data.sizes["row"]
    freq_channels = sample_data.DATA.shape[1]
    n_corr = sample_data.DATA.shape[2]

    # Calculate memory per baseline
    # For GPU: time samples × frequency channels × sizeof(float32) × 3 (data, flags, thresholds) × correlations
    corr_factor = 1  # Default to processing one correlation at a time

    # If we're combining correlations, we need to account for that
    if options.get("corr_to_process") and len(options["corr_to_process"]) > 1:
        # We're processing multiple correlations together
        corr_factor = len(options["corr_to_process"])

    # GPU memory per baseline
    gpu_mem_per_bl = (
        time_samples * freq_channels * 4 * 3 * corr_factor
    )  # 4 bytes for float32

    # System memory requirements
    # Base memory for the data and processing
    system_mem_per_bl = (
        time_samples * freq_channels * 8 * n_corr
    )  # 8 bytes for complex64

    # Additional memory for flag application if enabled
    if options.get("apply_flags", False):
        system_mem_per_bl += (
            time_samples * freq_channels * n_corr
        )  # 1 byte for bool flags

    # Additional memory for diagnostic plots if enabled
    if options.get("diagnostic_plots", False):
        # Plotting typically requires several copies of the data
        # For color maps, masked arrays, etc.
        plot_mem_factor = 4  # Conservative estimate
        system_mem_per_bl += time_samples * freq_channels * 4 * plot_mem_factor

    print(f"Memory estimates per baseline:")
    print(f"  - GPU: {gpu_mem_per_bl / (1024 * 1024):.2f} MB")
    print(f"  - System: {system_mem_per_bl / (1024 * 1024):.2f} MB")

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

    # Add a fixed upper limit to be safe
    max_batch_size = 50
    if bl_batch_size > max_batch_size:
        print(f"Limiting batch size to {max_batch_size} for safety")
        bl_batch_size = max_batch_size

    return bl_batch_size


def hunt_ms(ms_file, options):
    """The main function which reads the MS, then based on the sizes of each baseline, calculates
    the number of baselines which can be processed in the GPU. This is the top-level wrapper for reading and
    processing the data in the GPU."""

    print(f"\nPOOH is on the hunt in {ms_file}...")

    # Create output directory if diagnostic plots are enabled
    if options["diagnostic_plots"]:
        output_dir = options.get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)

    # Get antenna info from ANTENNA subtable
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

    # Get frequency axis for plotting
    freq_axis = spw_ds.CHAN_FREQ.values[0]

    # Get observation info
    obs_ds = xds_from_table(f"{ms_file}::OBSERVATION")[0]
    print("\n==== OBSERVATION INFO ====")
    print(f"Observer: {obs_ds.OBSERVER.values[0]}")
    print(f"Project: {obs_ds.PROJECT.values[0]}")
    print(f"Telescope: {obs_ds.TELESCOPE_NAME.values[0]}")

    # Get field info
    field_ds = xds_from_table(f"{ms_file}::FIELD")[0]
    print("\n==== FIELD INFO ====")
    print(f"Field names: {field_ds.NAME.values}")

    # Track overall statistics
    total_start_time = time.time()
    total_flagged = 0
    total_new_flags = 0
    total_visibilities = 0
    baselines_processed = 0
    baselines_skipped = 0

    # Get GPU information
    try:
        gpu_device = cuda.get_current_device()

        # Try different ways to get memory info
        try:
            gpu_mem = gpu_device.total_memory
        except AttributeError:
            try:
                # Alternative approach - try to get memory info from context
                gpu_mem = cuda.current_context().get_memory_info()[1]  # total memory
            except:
                # Another approach sometimes used
                mem_info = cuda.cuda.cudart().cudaMemGetInfo()
                gpu_mem = mem_info[1]  # total memory

        usable_mem = gpu_mem * 0.8  # Use 80% of GPU memory to be safe
        print(f"\nGPU memory: {gpu_mem / 1e9:.2f} GB (using {usable_mem / 1e9:.2f} GB)")
    except Exception as e:
        print(f"Warning: Could not determine GPU memory, assuming 8GB: {str(e)}")
        usable_mem = 8 * 1024 * 1024 * 1024 * 0.8  # Assume 8GB, use 80%

    # Get system memory info
    try:
        import psutil

        system_mem = psutil.virtual_memory()
        total_system_mem = system_mem.total
        available_system_mem = system_mem.available
        system_usable_mem = available_system_mem * options.get(
            "max_memory_usage", 0.8
        )  # Use configurable % of available memory
        print(
            f"System memory: {total_system_mem / 1e9:.2f} GB (available: {available_system_mem / 1e9:.2f} GB, using: {system_usable_mem / 1e9:.2f} GB)"
        )
    except Exception as e:
        print(f"Warning: Could not determine system memory, assuming 16GB: {str(e)}")
        system_usable_mem = 16 * 1024 * 1024 * 1024 * 0.8  # Assume 16GB, use 80%

    print("Getting field IDs...")
    try:
        # Extract directly from the FIELD subtable instead of main table
        field_ds = xds_from_table(f"{ms_file}::FIELD")[0]
        field_ids = np.arange(field_ds.sizes["row"])
        print(f"Found field IDs: {field_ids}")
    except Exception as e:
        print(f"Warning: Could not extract field IDs from MS: {str(e)}")

        print(f"Found field IDs: {field_ids}")

    # Process each field separately
    for field_id in field_ids:
        print(f"\n*** Processing Field {field_id} ***")

        # Get unique baselines for this field
        print(f"Getting unique baselines for field {field_id}...")
        baselines = []

        # Read ANTENNA1 and ANTENNA2 for this field
        taql_where = f"FIELD_ID={field_id}"
        xds_list = xds_from_ms(
            ms_file,
            columns=("ANTENNA1", "ANTENNA2"),
            taql_where=taql_where,
            chunks={"row": 50000},
        )

        # Extract unique baseline pairs
        for ds in xds_list:
            ant1, ant2 = dask.compute(ds.ANTENNA1.data, ds.ANTENNA2.data)
            for a1, a2 in zip(ant1, ant2):
                if (a1, a2) not in baselines:
                    baselines.append((a1, a2))

        print(f"Found {len(baselines)} unique baselines in field {field_id}")

        # Get a sample baseline to check data shape and estimate memory
        print("Checking data shape with a sample baseline...")
        sample_bl = baselines[0] if baselines else None

        if not sample_bl:
            print(f"No baselines found for field {field_id}. Skipping.")
            continue

        # Read a small sample to get data shape - use simpler query without LIMIT
        sample_taql = f"FIELD_ID={field_id} AND ANTENNA1={sample_bl[0]} AND ANTENNA2={sample_bl[1]}"
        try:
            sample_ds = xds_from_ms(ms_file, columns=("DATA",), taql_where=sample_taql)[
                0
            ]

            if sample_ds.sizes["row"] == 0:
                print(f"No data found for sample baseline {sample_bl}. Skipping field.")
                continue

            data_shape = sample_ds.DATA.shape
        except Exception as e:
            print(f"Error reading sample baseline: {str(e)}")
            print("Using a direct read without TaQL to get data shape...")
            # Try a different approach without complex TaQL
            try:
                # Just read the first few rows of the MS to get the shape
                simple_ds = xds_from_ms(ms_file, columns=("DATA",))[0]
                data_shape = simple_ds.DATA.shape
            except Exception as e2:
                print(f"Could not determine data shape: {str(e2)}")
                print("Skipping field.")
                continue

        print(f"Data shape: {data_shape}")

        # According to the MS printout, dimensions are (row, chan, corr)
        n_corr = data_shape[2]  # Correlation is the third axis (index 2)
        freq_channels = data_shape[1]  # Channels is the second axis (index 1)

        print(f"Number of correlations: {n_corr}")

        # Determine which correlations to process
        corr_to_process = options["corr_to_process"]
        if corr_to_process is None:
            # Default: process first and last correlation together
            if n_corr >= 2:
                corr_to_process = [0, n_corr - 1]
                print(f"Processing correlations 0 and {n_corr - 1} (default)")
            else:
                corr_to_process = [0]
                print("Processing correlation 0 (default)")
        else:
            print(f"Processing correlations: {corr_to_process}")

        # Get number of time samples (rows) for the baseline
        time_samples = sample_ds.sizes["row"]

        # Get a better estimate by checking the actual count for this baseline
        count_taql = f"FIELD_ID={field_id} AND ANTENNA1={sample_bl[0]} AND ANTENNA2={sample_bl[1]}"
        count_ds = xds_from_ms(ms_file, columns=("ANTENNA1",), taql_where=count_taql)[0]
        time_samples = count_ds.sizes["row"]

        # Calculate memory requirements
        # For each baseline: time samples × frequency channels × sizeof(float32) × 3 (data, flags, thresholds)
        # Each correlation is processed separately, so we account for 1 correlation at a time
        memory_per_bl_gpu = (
            time_samples * freq_channels * 4 * 3
        )  # 4 bytes for float32, 3 for data+flags+thresholds

        # System memory requirements
        memory_per_bl_cpu = (
            time_samples * freq_channels * 8 * n_corr
        )  # 8 bytes for complex64 data + flags

        # Additional memory if plots are being generated
        if options["diagnostic_plots"]:
            memory_per_bl_cpu += (
                time_samples * freq_channels * 4 * 3
            )  # Additional copies for plotting

        print(
            f"Typical baseline has {time_samples} time samples × {freq_channels} frequency channels"
        )
        print(
            f"Estimated memory per baseline: {memory_per_bl_gpu / (1024 * 1024):.2f} MB (GPU), {memory_per_bl_cpu / (1024 * 1024):.2f} MB (CPU)"
        )

        # Determine how many complete baselines can fit in GPU memory
        bl_per_batch_gpu = max(1, int(usable_mem // memory_per_bl_gpu))
        bl_per_batch_cpu = max(1, int(system_usable_mem // memory_per_bl_cpu))
        bl_per_batch = min(bl_per_batch_gpu, bl_per_batch_cpu) // 2

        print(
            f"Can process approximately {bl_per_batch} complete baselines in memory at once"
        )

        # # Limit batch size for safety
        # max_batch_size = 20  # Safety limit
        # if bl_per_batch > max_batch_size:
        #     print(f"Limiting batch size to {max_batch_size} for safety")
        #     bl_per_batch = max_batch_size

        # If a single baseline doesn't fit, calculate chunking factor
        if memory_per_bl_gpu > usable_mem:
            chunk_factor = int(np.ceil(memory_per_bl_gpu / usable_mem))
            chunk_size = time_samples // chunk_factor
            print(
                f"Single baseline requires chunking! Splitting into {chunk_factor} chunks"
            )
            print(f"Chunk size: {chunk_size} time samples")
        else:
            chunk_factor = 1
            chunk_size = time_samples

        # Process baselines in batches if possible
        if bl_per_batch > 1 and chunk_factor == 1:
            # We can fit multiple baselines in GPU memory
            # Process in batches of size bl_per_batch
            for i in range(0, len(baselines), bl_per_batch):
                batch = baselines[i : i + bl_per_batch]
                batch_size = len(batch)

                print(
                    f"\nProcessing batch {i // bl_per_batch + 1}/{(len(baselines) + bl_per_batch - 1) // bl_per_batch}: {batch_size} baselines"
                )

                # OPTIMIZED READING: Read all baselines in this batch with a single query
                # Create TaQL filter for all baselines in the batch
                baseline_clauses = []
                for ant1, ant2 in batch:
                    baseline_clauses.append(f"(ANTENNA1={ant1} AND ANTENNA2={ant2})")

                taql_where = (
                    f"FIELD_ID={field_id} AND ({' OR '.join(baseline_clauses)})"
                )

                # Read all data in a single query
                batch_ds_list = xds_from_ms(
                    ms_file,
                    columns=("DATA", "FLAG", "ANTENNA1", "ANTENNA2"),
                    taql_where=taql_where,
                )

                # Dictionary to store baseline data
                baseline_data = {}

                # Process each dataset
                for ds in batch_ds_list:
                    # Materialize data, flags, and antenna indices in one operation
                    ant1, ant2, data, flags = dask.compute(
                        ds.ANTENNA1.data, ds.ANTENNA2.data, ds.DATA.data, ds.FLAG.data
                    )

                    # Organize data by baseline
                    for i, (a1, a2) in enumerate(zip(ant1, ant2)):
                        bl = (a1, a2)

                        # Skip if this baseline is not in our batch
                        if bl not in batch:
                            continue

                        # Initialize storage for this baseline if needed
                        if bl not in baseline_data:
                            baseline_data[bl] = {"data": [], "flags": [], "indices": []}

                        # Store this row's data
                        baseline_data[bl]["data"].append(data[i])
                        baseline_data[bl]["flags"].append(flags[i])
                        baseline_data[bl]["indices"].append(i)

                # Create materialized data objects for GPU processing
                valid_baseline_data = {}

                for bl in batch:
                    if bl not in baseline_data or not baseline_data[bl]["data"]:
                        if options["verbose"]:
                            print(f"No data found for baseline {bl}. Skipping.")
                        baselines_skipped += 1
                        continue

                    # Combine data for this baseline
                    bl_data = np.array(baseline_data[bl]["data"])
                    bl_flags = np.array(baseline_data[bl]["flags"])

                    # Check if baseline is completely flagged
                    if np.all(bl_flags):
                        if options["verbose"]:
                            print(f"Baseline {bl} is completely flagged. Skipping.")
                        baselines_skipped += 1
                        continue

                    # Create materialized data object
                    class MaterializedData:
                        def __init__(self, data, flags, row_size):
                            self.DATA = data
                            self.FLAG = flags
                            self.sizes = {"row": row_size}

                    materialized_bl_data = MaterializedData(
                        bl_data, bl_flags, len(bl_data)
                    )

                    valid_baseline_data[bl] = materialized_bl_data

                # OPTIMIZED GPU PROCESSING: Process all baselines with minimal GPU data transfers
                if valid_baseline_data:
                    # Use our optimized batch processor
                    batch_results = process_baselines_batch_gpu(
                        valid_baseline_data,
                        field_id,
                        corr_to_process,
                        options,
                        freq_axis,
                    )

                    # Process results and collect statistics
                    for bl, flags in batch_results.items():
                        try:
                            # Get original flags for comparison
                            original_flags = valid_baseline_data[bl].FLAG

                            # Count existing and new flags
                            existing_count = np.sum(original_flags)
                            total_flags = np.sum(flags)
                            new_count = total_flags - existing_count

                            # Update statistics
                            total_flagged += existing_count
                            total_new_flags += new_count
                            total_visibilities += np.prod(original_flags.shape)

                            baselines_processed += 1
                        except Exception as e:
                            print(
                                f"Error processing results for baseline {bl}: {str(e)}"
                            )
                            if options["verbose"]:
                                import traceback

                                traceback.print_exc()
                            continue

                    # Write flags to MS if requested
                    if options["apply_flags"]:
                        # First read back the original datasets to preserve metadata
                        try:
                            # Create TaQL for the baselines we processed
                            processed_bl_clauses = []
                            for bl in batch_results.keys():
                                processed_bl_clauses.append(
                                    f"(ANTENNA1={bl[0]} AND ANTENNA2={bl[1]})"
                                )

                            if not processed_bl_clauses:
                                print("No baselines to write flags for in this batch.")
                                continue

                            taql_where = f"FIELD_ID={field_id} AND ({' OR '.join(processed_bl_clauses)})"

                            # Read data back to get correct dataset structure
                            orig_ds_list = xds_from_ms(
                                ms_file,
                                columns=("FLAG", "ANTENNA1", "ANTENNA2"),
                                taql_where=taql_where,
                            )

                            # Create updated datasets
                            updated_datasets = []

                            for ds in orig_ds_list:
                                # Get antenna info to identify baseline
                                ant1, ant2 = dask.compute(
                                    ds.ANTENNA1.data, ds.ANTENNA2.data
                                )

                                # Create a new FLAG array
                                orig_flags = ds.FLAG.data  # This is a dask array
                                new_flags = orig_flags.compute()  # Convert to numpy

                                # Update flags for rows matching processed baselines
                                for row_idx, (a1, a2) in enumerate(zip(ant1, ant2)):
                                    bl = (a1, a2)
                                    if bl in batch_results:
                                        # Get new flags for this baseline
                                        bl_flags = batch_results[bl]
                                        # Update if in range
                                        if row_idx < bl_flags.shape[0]:
                                            new_flags[row_idx] = np.logical_or(
                                                new_flags[row_idx], bl_flags[row_idx]
                                            )

                                # Convert back to dask array with same chunking as original
                                import dask.array as da

                                new_flags_dask = da.from_array(
                                    new_flags, chunks=orig_flags.chunks
                                )

                                # Create updated dataset with new flags
                                updated_ds = ds.assign(
                                    FLAG=(ds.FLAG.dims, new_flags_dask)
                                )
                                updated_datasets.append(updated_ds)

                            # Write back to MS
                            if updated_datasets:
                                try:
                                    print(
                                        f"Writing flags for batch {i // bl_per_batch + 1}..."
                                    )
                                    write_back = xds_to_table(
                                        updated_datasets, ms_file, ["FLAG"]
                                    )
                                    dask.compute(write_back)
                                    print(
                                        f"Successfully wrote flag updates for batch {i // bl_per_batch + 1}"
                                    )
                                except Exception as e:
                                    print(f"Error writing flags back to MS: {str(e)}")
                                    import traceback

                                    traceback.print_exc()
                        except Exception as e:
                            print(f"Error preparing flag updates: {str(e)}")
                            import traceback

                            traceback.print_exc()
                else:
                    print("No valid baselines found in this batch. Skipping.")

                # Clean up to free memory
                batch_ds_list = None
                baseline_data = None
                valid_baseline_data = None
                batch_results = None

                # Force garbage collection
                import gc

                gc.collect()
        else:
            # We need to process baselines one at a time, possibly in chunks
            for bl_idx, bl in enumerate(tqdm(baselines, desc="Baselines")):
                if options["verbose"]:
                    print(
                        f"\nProcessing baseline {bl_idx + 1}/{len(baselines)}: Antennas {bl[0]}-{bl[1]}"
                    )

                try:
                    # Read this baseline with direct TaQL
                    taql_where = (
                        f"FIELD_ID={field_id} AND ANTENNA1={bl[0]} AND ANTENNA2={bl[1]}"
                    )

                    bl_ds_list = xds_from_ms(
                        ms_file, columns=("DATA", "FLAG"), taql_where=taql_where
                    )

                    # Check if we got any data
                    if not bl_ds_list or bl_ds_list[0].sizes["row"] == 0:
                        if options["verbose"]:
                            print(f"No data found for baseline {bl}. Skipping.")
                        baselines_skipped += 1
                        continue

                    # Materialized data from first dataset
                    bl_data, bl_flags = dask.compute(
                        bl_ds_list[0].DATA.data, bl_ds_list[0].FLAG.data
                    )

                    # Check if completely flagged
                    if np.all(bl_flags):
                        if options["verbose"]:
                            print(f"Baseline {bl} is completely flagged. Skipping.")
                        baselines_skipped += 1
                        continue

                    # Create materialized data object
                    class MaterializedData:
                        def __init__(self, data, flags, row_size):
                            self.DATA = data
                            self.FLAG = flags
                            self.sizes = {"row": row_size}

                    materialized_bl_data = MaterializedData(
                        bl_data, bl_flags, bl_data.shape[0]
                    )

                    # Initialize tracking variables
                    baseline_existing_flags = 0
                    baseline_new_flags = 0

                    # If we need to chunk this baseline
                    if chunk_factor > 1:
                        chunk_flags = []
                        # Create streams for parallel chunk processing (GPU only)
                        if GPU_AVAILABLE:
                            chunk_streams = [cuda.stream() for _ in range(chunk_factor)]
                        else:
                            chunk_streams = [None for _ in range(chunk_factor)]
                        chunk_results = []
                        chunk_infos = []

                        # Launch all chunk processing in parallel
                        for chunk, stream in enumerate(chunk_streams):
                            start_idx = chunk * chunk_size
                            end_idx = min((chunk + 1) * chunk_size, time_samples)

                            if options["verbose"]:
                                print(
                                    f"Processing chunk {chunk + 1}/{chunk_factor}: rows {start_idx}-{end_idx}"
                                )

                            # Extract chunk of data
                            chunk_data = bl_data[start_idx:end_idx]
                            chunk_flags = bl_flags[start_idx:end_idx]

                            # Create chunk data object
                            chunk_obj = MaterializedData(
                                chunk_data, chunk_flags, end_idx - start_idx
                            )

                            # Process this chunk asynchronously
                            chunk_info = f"chunk {chunk + 1}/{chunk_factor}"
                            chunk_infos.append((start_idx, end_idx, chunk_info))

                            result = process_baseline_async(
                                chunk_obj,
                                bl,
                                field_id,
                                corr_to_process,
                                options,
                                freq_axis,
                                total_flagged,
                                total_visibilities,
                                stream,
                                chunk_info=chunk_info,
                                output_dir=options.get("output_dir", "outputs"),
                            )
                            chunk_results.append(result)

                        # Wait for all chunks to complete (GPU only)
                        if GPU_AVAILABLE:
                            cuda.synchronize()

                        # Process results
                        all_chunk_flags = None

                        for (start_idx, end_idx, chunk_info), result in zip(
                            chunk_infos, chunk_results
                        ):
                            chunk_flags_result, chunk_existing, chunk_new = result

                            # Update statistics
                            baseline_existing_flags += chunk_existing
                            baseline_new_flags += chunk_new

                            # Save chunk flags for combining
                            if chunk_flags_result is not None:
                                chunk_flags.append(
                                    (start_idx, end_idx, chunk_flags_result)
                                )

                        # Combine chunk flags if needed
                        if chunk_flags:
                            # Initialize flags array for full baseline
                            all_chunk_flags = np.zeros(
                                (bl_data.shape[0], bl_data.shape[1], bl_data.shape[2]),
                                dtype=bool,
                            )

                            # Fill in flags from each chunk
                            for start_idx, end_idx, flags in chunk_flags:
                                all_chunk_flags[start_idx:end_idx, :, :] = flags

                        # Clean up chunk streams
                        for stream in chunk_streams:
                            stream.synchronize()
                    else:
                        # Process the whole baseline at once
                        all_chunk_flags, baseline_existing, baseline_new = (
                            process_baseline_async(
                                materialized_bl_data,
                                bl,
                                field_id,
                                corr_to_process,
                                options,
                                freq_axis,
                                total_flagged,
                                total_visibilities,
                                output_dir=options.get("output_dir", "outputs"),
                            )
                        )

                        # Update statistics
                        baseline_existing_flags = baseline_existing
                        baseline_new_flags = baseline_new

                    # Update overall statistics
                    total_flagged += baseline_existing_flags
                    total_new_flags += baseline_new_flags
                    baselines_processed += 1

                    # Write flags to MS if requested and we have flags to write
                    if options["apply_flags"] and all_chunk_flags is not None:
                        try:
                            # Read the original dataset again to get correct structure
                            orig_ds_list = xds_from_ms(
                                ms_file, columns=("FLAG",), taql_where=taql_where
                            )

                            if orig_ds_list:
                                # Get the original FLAG column's structure
                                orig_ds = orig_ds_list[0]
                                orig_flags = orig_ds.FLAG

                                # CHANGE THESE LINES:
                                # From:
                                # import dask.array as da
                                # new_flags_dask = da.from_array(all_chunk_flags, chunks=orig_flags.data.chunks)
                                # updated_ds = orig_ds.assign(FLAG=(orig_flags.dims, new_flags_dask))

                                # To:
                                # Get original flags as numpy array
                                orig_flags_array = orig_ds.FLAG.data.compute()
                                # Combine original flags with new flags
                                combined_flags = np.logical_or(
                                    orig_flags_array, all_chunk_flags
                                )
                                # Convert back to dask array
                                new_flags_dask = da.from_array(
                                    combined_flags, chunks=orig_flags.data.chunks
                                )
                                # Create updated dataset
                                updated_ds = orig_ds.assign(
                                    FLAG=(orig_flags.dims, new_flags_dask)
                                )

                                # Write back to MS
                                try:
                                    print(f"Writing flags for baseline {bl}...")
                                    write_back = xds_to_table(
                                        [updated_ds], ms_file, ["FLAG"]
                                    )
                                    dask.compute(write_back)
                                except Exception as e:
                                    print(
                                        f"Error writing flags for baseline {bl}: {str(e)}"
                                    )
                                    import traceback

                                    traceback.print_exc()
                        except Exception as e:
                            print(
                                f"Error preparing flag updates for baseline {bl}: {str(e)}"
                            )
                            import traceback

                            traceback.print_exc()

                    # Clean up to free memory
                    bl_ds_list = None
                    bl_data = None
                    bl_flags = None
                    materialized_bl_data = None
                    all_chunk_flags = None
                    chunk_flags = None
                    chunk_results = None

                    # Force garbage collection
                    import gc

                    gc.collect()

                except Exception as e:
                    print(f"Error processing baseline {bl}: {str(e)}")
                    if options["verbose"]:
                        import traceback

                        traceback.print_exc()
                    continue

    # Calculate overall statistics
    total_time = time.time() - total_start_time
    overall_percent_flagged = (
        100 * (total_flagged + total_new_flags) / total_visibilities
        if total_visibilities > 0
        else 0
    )
    new_percent_flagged = (
        100 * total_new_flags / total_visibilities if total_visibilities > 0 else 0
    )

    # Prepare results dictionary
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
    """Process a single baseline or chunk of a baseline asynchronously using CUDA streams."""

    # Just use pixels for time axis
    time_axis = np.arange(bl_data.sizes["row"])
    time_res = 1.0

    # Initialize flags array if we need to return flags
    if options["apply_flags"]:
        # CRITICAL: Start with original flags to preserve flags in correlations not being processed
        combined_flags = bl_data.FLAG.copy()
    else:
        combined_flags = None

    # Track existing and new flags
    existing_flag_count = 0
    new_flag_count = 0

    # Store diagnostic data for later plot generation
    diagnostic_data = []

    if len(corr_to_process) == 1:
        # Process a single correlation
        corr_idx = corr_to_process[0]

        # Extract data - MS has shape (row, chan, corr)
        vis_data = bl_data.DATA[:, :, corr_idx]
        existing_flags = bl_data.FLAG[:, :, corr_idx]
        existing_flag_count += np.sum(existing_flags)

        # Calculate amplitude respecting existing flags
        # For flagged points, we'll just set them to 0
        masked_data = np.ma.masked_array(vis_data, mask=existing_flags)
        vis_amp = np.abs(masked_data.filled(0))

        # Skip if all data is flagged
        if np.all(existing_flags):
            if options["verbose"]:
                print(
                    f"All data is already flagged for baseline {bl} corr {corr_idx} {chunk_info}. Skipping."
                )
            if options["apply_flags"]:
                combined_flags[:, :, corr_idx] = existing_flags
            return combined_flags, existing_flag_count, 0

        # Run SumThreshold with stream
        new_flags, _ = sumthreshold_gpu(
            vis_amp,
            existing_flags,
            f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Pol {corr_idx} {chunk_info}",
            combinations=options["combinations"],
            sigma_factor=options["sigma_factor"],
            rho=options["rho"],
            diagnostic_plots=False,
            stream=stream,
        )

        # Combine with existing flags (logical OR)
        combined_flags_corr = np.logical_or(existing_flags, new_flags)

        # Count newly added flags
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

        # Store flags for this correlation if we need to apply them
        if options["apply_flags"]:
            combined_flags[:, :, corr_idx] = combined_flags_corr

    elif (
        len(corr_to_process) == 2
        and corr_to_process[0] == 0
        and corr_to_process[-1] == bl_data.DATA.shape[2] - 1
    ):
        # Sum first and last correlation (typical for Stokes I approximation)
        # Get data and flags for both correlations
        vis_data_0 = bl_data.DATA[:, :, corr_to_process[0]]
        vis_data_1 = bl_data.DATA[:, :, corr_to_process[1]]
        existing_flags_0 = bl_data.FLAG[:, :, corr_to_process[0]]
        existing_flags_1 = bl_data.FLAG[:, :, corr_to_process[1]]

        # Count existing flags
        existing_flag_count += np.sum(existing_flags_0) + np.sum(existing_flags_1)

        # For flagged points in either correlation, use 0 for the sum
        masked_data_0 = np.ma.masked_array(vis_data_0, mask=existing_flags_0).filled(0)
        masked_data_1 = np.ma.masked_array(vis_data_1, mask=existing_flags_1).filled(0)

        # Sum the data
        vis_sum = masked_data_0 + masked_data_1
        vis_amp = np.abs(vis_sum)

        # Combine existing flags from both correlations
        existing_flags_combined = np.logical_or(existing_flags_0, existing_flags_1)

        # Skip if all data is flagged
        if np.all(existing_flags_combined):
            if options["verbose"]:
                print(
                    f"All data is already flagged for baseline {bl} combined corrs {chunk_info}. Skipping."
                )
            if options["apply_flags"]:
                combined_flags[:, :, corr_to_process[0]] = existing_flags_0
                combined_flags[:, :, corr_to_process[1]] = existing_flags_1
            return combined_flags, existing_flag_count, 0

        # Run SumThreshold using the specified stream
        new_flags, _ = sumthreshold_gpu(
            vis_amp,
            existing_flags_combined,
            f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Combined Pols {chunk_info}",
            combinations=options["combinations"],
            sigma_factor=options["sigma_factor"],
            rho=options["rho"],
            diagnostic_plots=False,
            stream=stream,
        )

        # Combine with existing flags
        combined_flags_corr = np.logical_or(existing_flags_combined, new_flags)

        # Count newly added flags
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

        # Update total visibilities
        total_visibilities += vis_amp.size * len(corr_to_process)

        # Store flags for both correlations if we need to apply them
        if options["apply_flags"]:
            for corr_idx in corr_to_process:
                combined_flags[:, :, corr_idx] = combined_flags_corr

    else:
        # Process each correlation separately
        for corr_idx in corr_to_process:
            # Extract data - MS has shape (row, chan, corr)
            vis_data = bl_data.DATA[:, :, corr_idx]
            existing_flags = bl_data.FLAG[:, :, corr_idx]
            existing_flag_count += np.sum(existing_flags)

            # Calculate amplitude respecting existing flags
            masked_data = np.ma.masked_array(vis_data, mask=existing_flags)
            vis_amp = np.abs(masked_data.filled(0))

            # Skip if all data is flagged
            if np.all(existing_flags):
                if options["verbose"]:
                    print(
                        f"All data is already flagged for baseline {bl} corr {corr_idx} {chunk_info}. Skipping."
                    )
                if options["apply_flags"]:
                    combined_flags[:, :, corr_idx] = existing_flags
                continue

            # Run SumThreshold with the specified stream
            new_flags, _ = sumthreshold_gpu(
                vis_amp,
                existing_flags,
                f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Pol {corr_idx} {chunk_info}",
                combinations=options["combinations"],
                sigma_factor=options["sigma_factor"],
                rho=options["rho"],
                diagnostic_plots=False,
                stream=stream,
            )

            # Combine with existing flags
            combined_flags_corr = np.logical_or(existing_flags, new_flags)

            # Count newly added flags
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

            # Store flags for this correlation if we need to apply them
            if options["apply_flags"]:
                combined_flags[:, :, corr_idx] = combined_flags_corr

    # Generate diagnostic plots after all processing is complete
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

    # Return the flags and flag counts
    return combined_flags, existing_flag_count, new_flag_count


# 2. Now, modify the process_baselines_batch_gpu function to also generate diagnostic plots


def process_baselines_batch_gpu(
    baseline_data, field_id, corr_to_process, options, freq_axis
):
    """
    Process a batch of baselines with minimal GPU data transfers.

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

    # Extract baselines to process
    baselines = list(baseline_data.keys())
    print(
        f"Processing batch of {len(baselines)} baselines with optimized GPU transfers..."
    )

    # Results dictionary
    results = {}

    # Dictionary to store diagnostic data
    diagnostic_data = {} if options["diagnostic_plots"] else None

    # First, pre-calculate thresholds for all baselines in the batch on CPU
    print("Pre-calculating thresholds for all baselines...")

    # Dictionary to store threshold arrays
    threshold_arrays = {}

    # Calculate thresholds for all baselines first (CPU-intensive part)
    for bl in baselines:
        bl_data = baseline_data[bl]

        # Process each correlation type as needed
        if (
            len(corr_to_process) == 2
            and corr_to_process[0] == 0
            and corr_to_process[-1] == bl_data.DATA.shape[2] - 1
        ):
            # Sum first and last correlation (typical for Stokes I approximation)
            # Get data and flags for both correlations
            vis_data_0 = bl_data.DATA[:, :, corr_to_process[0]]
            vis_data_1 = bl_data.DATA[:, :, corr_to_process[1]]
            flags_0 = bl_data.FLAG[:, :, corr_to_process[0]]
            flags_1 = bl_data.FLAG[:, :, corr_to_process[1]]

            # For flagged points in either correlation, use 0 for the sum
            masked_data_0 = np.ma.masked_array(vis_data_0, mask=flags_0).filled(0)
            masked_data_1 = np.ma.masked_array(vis_data_1, mask=flags_1).filled(0)

            # Sum the data
            vis_sum = masked_data_0 + masked_data_1
            vis_amp = np.abs(vis_sum)

            # Combine existing flags from both correlations
            existing_flags_combined = np.logical_or(flags_0, flags_1)

            # Calculate channel-based statistics for thresholds
            channel_medians = np.zeros(vis_amp.shape[1], dtype=np.float32)
            channel_stds = np.zeros(vis_amp.shape[1], dtype=np.float32)

            # Calculate statistics while respecting existing flags
            for j in range(vis_amp.shape[1]):
                # Get data for this channel, masking existing flags
                channel_data = vis_amp[:, j]
                mask = existing_flags_combined[:, j]
                unflagged_data = channel_data[~mask]

                if len(unflagged_data) > 0:
                    # Calculate statistics only on unflagged data
                    channel_medians[j] = np.median(unflagged_data)
                    channel_stds[j] = np.std(unflagged_data)
                else:
                    # If all data is flagged in this channel, use reasonable defaults
                    # Use median and std of entire dataset or some default values
                    all_unflagged = vis_amp[~existing_flags_combined]
                    if len(all_unflagged) > 0:
                        channel_medians[j] = np.median(all_unflagged)
                        channel_stds[j] = np.std(all_unflagged)
                    else:
                        # If everything is flagged, use reasonable defaults
                        channel_medians[j] = 0.0
                        channel_stds[j] = 1.0

            # Store thresholds for this baseline
            threshold_arrays[bl] = {
                "base_thresholds": channel_medians
                + options["sigma_factor"] * channel_stds,
                "vis_amp": vis_amp,
                "existing_flags": existing_flags_combined,
                "mode": "combined",
            }

        else:
            # Process each correlation separately
            for corr_idx in corr_to_process:
                # Extract data - MS has shape (row, chan, corr)
                vis_data = bl_data.DATA[:, :, corr_idx]
                existing_flags = bl_data.FLAG[:, :, corr_idx]

                # Calculate amplitude respecting existing flags
                masked_data = np.ma.masked_array(vis_data, mask=existing_flags)
                vis_amp = np.abs(masked_data.filled(0))

                # Calculate channel-based statistics for thresholds
                channel_medians = np.zeros(vis_amp.shape[1], dtype=np.float32)
                channel_stds = np.zeros(vis_amp.shape[1], dtype=np.float32)

                # Calculate statistics while respecting existing flags
                for j in range(vis_amp.shape[1]):
                    # Get data for this channel, masking existing flags
                    channel_data = vis_amp[:, j]
                    mask = existing_flags[:, j]
                    unflagged_data = channel_data[~mask]

                    if len(unflagged_data) > 0:
                        # # Calculate statistics only on unflagged data
                        channel_medians[j] = np.median(unflagged_data)
                        channel_stds[j] = np.std(unflagged_data)
                        #  channel_medians[j] = np.median(channel_data)
                        #  channel_stds[j] = np.std(channel_data)
                    else:
                        # If all data is flagged in this channel, use reasonable defaults
                        # Use median and std of entire dataset or some default values
                        all_unflagged = vis_amp[~existing_flags]
                        if len(all_unflagged) > 0:
                            channel_medians[j] = np.median(all_unflagged)
                            channel_stds[j] = np.std(all_unflagged)
                        else:
                            # If everything is flagged, use reasonable defaults
                            channel_medians[j] = 0.0
                            channel_stds[j] = 1.0

                # Store for this correlation - make sure we use a tuple of (baseline, corr_idx)
                # Create a unique key for this baseline-correlation combination
                baseline_corr_key = (
                    bl[0],
                    bl[1],
                    corr_idx,
                )  # Use a flat tuple to avoid nested tuples

                threshold_arrays[baseline_corr_key] = {
                    "base_thresholds": channel_medians
                    + options["sigma_factor"] * channel_stds,
                    "vis_amp": vis_amp,
                    "existing_flags": existing_flags,
                    "mode": "single",
                    "corr_idx": corr_idx,
                    "bl": bl,  # Store the original baseline tuple
                }

    print(f"Threshold calculation completed for {len(threshold_arrays)} datasets")

    # Debug: Print some keys to understand their structure
    if threshold_arrays:
        sample_keys = list(threshold_arrays.keys())[:3]  # Print first 3 keys
        print("DEBUG - Sample keys in threshold_arrays:")
        for k in sample_keys:
            print(f"  Key: {k}, Type: {type(k)}")

    # Now process all baselines with a single GPU allocation (if available)
    # Create a CUDA stream (GPU only)
    if GPU_AVAILABLE:
        stream = cuda.stream()
    else:
        stream = None

    # Process each baseline/correlation with the pre-calculated thresholds
    for key, threshold_data in threshold_arrays.items():
        # Handle different key types safely
        try:
            if threshold_data["mode"] == "single":
                # Single correlation mode - extract data from the stored information
                corr_idx = threshold_data["corr_idx"]
                bl = threshold_data["bl"]  # Get the original baseline tuple
                info_string = (
                    f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Pol {corr_idx}"
                )
            else:
                # Combined correlations mode
                bl = key  # In this case, key should be the baseline tuple
                info_string = (
                    f"Field {field_id}, Baseline {bl[0]}-{bl[1]}, Combined Pols"
                )

            # Get data
            vis_amp = threshold_data["vis_amp"]
            existing_flags = threshold_data["existing_flags"]
            base_thresholds = threshold_data["base_thresholds"]

            # Run the GPU flagger using existing function but with pre-calculated thresholds
            flags, _ = sumthreshold_gpu(
                vis_amp,
                existing_flags,
                info_string,
                combinations=options["combinations"],
                sigma_factor=None,  # Not needed since we pre-calculated thresholds
                rho=options["rho"],
                diagnostic_plots=False,
                stream=stream,
                precalculated_thresholds=base_thresholds,  # Pass pre-calculated thresholds
            )

            # Combine with existing flags
            combined_flags_corr = np.logical_or(existing_flags, flags)

            # Store diagnostic data if requested
            if options["diagnostic_plots"]:
                mode = threshold_data["mode"]
                if mode == "single":
                    corr_idx = threshold_data["corr_idx"]
                    bl = threshold_data["bl"]
                    diag_key = (bl, corr_idx)
                else:
                    diag_key = (bl, "combined")

                # Initialize diagnostic data dictionary if needed
                if diagnostic_data is None:
                    diagnostic_data = {}

                # Store diagnostic data
                diagnostic_data[diag_key] = {
                    "vis_amp": vis_amp.copy(),
                    "combined_flags": combined_flags_corr.copy(),
                    "existing_flags": existing_flags.copy(),
                    "corr_idx": "combined" if mode == "combined" else corr_idx,
                    "bl": bl,
                    "field_id": field_id,
                }

            # Store the results
            mode = threshold_data["mode"]
            if mode == "combined":
                # For combined mode, create full FLAG array
                if bl not in results:
                    # Initialize full flag array
                    bl_data = baseline_data[bl]
                    full_flags = bl_data.FLAG.copy()

                    # Update all correlations with the same flags
                    for corr_idx in corr_to_process:
                        # CHANGE THIS LINE:
                        # From: full_flags[:, :, corr_idx] = combined_flags_corr
                        # To:
                        full_flags[:, :, corr_idx] = np.logical_or(
                            full_flags[:, :, corr_idx], combined_flags_corr
                        )

                    # Store
                    results[bl] = full_flags
            else:
                # Single correlation mode
                corr_idx = threshold_data["corr_idx"]
                orig_bl = threshold_data["bl"]  # Get the original baseline tuple

                if orig_bl not in results:
                    # Initialize full flag array
                    bl_data = baseline_data[orig_bl]
                    full_flags = bl_data.FLAG.copy()
                    results[orig_bl] = full_flags

                # Update this correlation's flags
                results[orig_bl][:, :, corr_idx] = np.logical_or(
                    results[orig_bl][:, :, corr_idx], combined_flags_corr
                )

        except Exception as e:
            print(f"Error processing key {key}: {str(e)}")
            import traceback

            traceback.print_exc()

    # Generate diagnostic plots after all processing is complete
    if options["diagnostic_plots"] and diagnostic_data:
        print(f"Generating {len(diagnostic_data)} diagnostic plots...")
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
    print(f"Batch processing completed in {total_time:.2f} seconds")

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
    """Generate a simple diagnostic plot showing flagged and unflagged data without downsampling"""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert corr_idx to string representation if it's not already a string
    corr_str = str(corr_idx) if isinstance(corr_idx, (int, float)) else corr_idx

    # Create the output filename
    filename = os.path.join(
        output_dir, f"pooh_field{field_id}_bl{bl[0]}-{bl[1]}_pol{corr_str}.png"
    )

    try:
        print(
            f"Generating plot for bl {bl[0]}-{bl[1]} pol {corr_str}, shape: {vis_amp.shape}"
        )

        # Check if the array is too large to plot efficiently
        total_pixels = vis_amp.size
        if total_pixels > 1000000:  # 1 million pixels
            print(
                f"Warning: Large data array ({total_pixels} pixels) may consume significant memory"
            )

        # Use smaller figure size and low DPI for memory efficiency
        plt.figure(figsize=(10, 5), dpi=80)

        # Left panel: Raw dynamic spectra (unflagged)
        plt.subplot(1, 2, 1)

        # Plot the original unflagged amplitude
        plt.imshow(
            vis_amp,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=100,
        )

        # Simple colorbar
        plt.colorbar(label="Amplitude", shrink=0.8)
        plt.xlabel("Frequency")
        plt.ylabel("Time")
        plt.title("Raw Dynamic Spectra")

        # Right panel: Flagged dynamic spectra
        plt.subplot(1, 2, 2)

        # Create masked array and replace flagged values with NaN for cleaner display
        masked_data = np.copy(vis_amp)
        masked_data[~combined_flags] = np.nan

        # Plot the masked data
        plt.imshow(
            masked_data,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=100,
        )
        plt.colorbar(label="Amplitude", shrink=0.8)
        plt.xlabel("Frequency")
        plt.ylabel("Time")
        plt.title("Flagged Dynamic Spectra")

        # Add percentage info
        percent_flagged = 100 * np.sum(combined_flags) / combined_flags.size
        percent_new = (
            100
            * (np.sum(combined_flags) - np.sum(existing_flags))
            / combined_flags.size
        )

        plt.suptitle(
            f"BL {bl[0]}-{bl[1]}, Pol {corr_str}, Field {field_id}\n"
            f"Total: {percent_flagged:.1f}% flagged, New: {percent_new:.1f}%"
        )

        # Save with appropriate parameters
        plt.tight_layout()
        plt.savefig(
            filename,
            dpi=80,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=False,
            facecolor="white",
        )

    except Exception as e:
        print(f"Error generating diagnostic plot: {str(e)}")
    finally:
        # Clean up to free memory
        plt.close("all")
        # Force garbage collection
        import gc

        gc.collect()


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

#     print(" Parrallelized Optimized Outlier Hunter: POOH, Developed by Arpan Pal at NCRA-TIFR in 2025")
#     print("Pooh is my first cat. He specializes in collecting snakes and birds, which he thoughtfully deposits on my bed. I write this code in return of those gifts.")
#     # Process the MS
#     results = hunt_ms(args.ms_path, options)

#     print("\nPOOH Flagging Summary:")
#     print(f"Measurement Set: {args.ms_path}")
#     print(f"Total processing time: {results['total_processing_time']:.2f} seconds")
#     print(f"Overall flagging percentage: {results['overall_percent_flagged']:.2f}%")
#     print("Meow! Flagging complete.")


# if __name__ == "__main__":
#     main()
