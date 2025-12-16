#include "data_collection.h"

CollectedData collect_data_points(
    py::array_t<std::complex<float>> data,
    py::array_t<bool> flags,
    py::array_t<double> uv_distances,
    py::array_t<float> weight,
    std::vector<int> spw_row_indices,
    std::vector<int> corr_indices
) {
    // Get array buffers
    auto data_buf = data.request();
    auto flags_buf = flags.request();
    auto uv_buf = uv_distances.request();
    auto weight_buf = weight.request();

    // Validate shapes
    if (data_buf.ndim != 3 || flags_buf.ndim != 3) {
        throw std::runtime_error("Data and flags must be 3D arrays");
    }
    if (uv_buf.ndim != 2) {
        throw std::runtime_error("UV distances must be 2D array");
    }

    size_t n_channels = data_buf.shape[1];
    size_t n_corr = data_buf.shape[2];

    // Access data pointers
    auto data_ptr = static_cast<std::complex<float>*>(data_buf.ptr);
    auto flags_ptr = static_cast<bool*>(flags_buf.ptr);
    auto uv_ptr = static_cast<double*>(uv_buf.ptr);
    auto weight_ptr = static_cast<float*>(weight_buf.ptr);

    // Result storage
    CollectedData result;

    // Exact translation of Python code (lines 488-520):
    // for row_idx in np.where(spw_mask)[0]:
    //     for chan_idx in range(n_channels):
    //         # Check if this channel is already fully flagged
    //         if np.all(flags[row_idx, chan_idx, :]):
    //             continue
    //
    //         # Get UV distance for this channel
    //         uv_dist = uv_distances[row_idx, chan_idx]
    //
    //         # Skip invalid UV points
    //         if uv_dist <= 0:
    //             continue
    //
    //         # Calculate amplitude as average of selected correlations
    //         valid_corrs = []
    //         amps = []
    //         for corr in corr_indices:
    //             if not flags[row_idx, chan_idx, corr] and np.abs(data[row_idx, chan_idx, corr]) > 0:
    //                 valid_corrs.append(corr)
    //                 amps.append(np.abs(data[row_idx, chan_idx, corr]))
    //
    //         # Skip if no valid correlations
    //         if not valid_corrs:
    //             continue
    //
    //         # Calculate mean amplitude and weight
    //         amp = np.mean(amps)
    //         w = np.mean(weight[row_idx])
    //
    //         all_uv.append(uv_dist)
    //         all_amp.append(amp)
    //         all_weights.append(w)
    //         point_indices.append((row_idx, chan_idx))

    for (int row_idx : spw_row_indices) {
        for (size_t chan_idx = 0; chan_idx < n_channels; ++chan_idx) {
            // Check if this channel is already fully flagged
            bool all_flagged = true;
            for (size_t corr = 0; corr < n_corr; ++corr) {
                size_t flag_idx = row_idx * n_channels * n_corr + chan_idx * n_corr + corr;
                if (!flags_ptr[flag_idx]) {
                    all_flagged = false;
                    break;
                }
            }
            if (all_flagged) continue;

            // Get UV distance for this channel
            size_t uv_idx = row_idx * n_channels + chan_idx;
            double uv_dist = uv_ptr[uv_idx];

            // Skip invalid UV points
            if (uv_dist <= 0.0) continue;

            // Collect real and imaginary parts of selected correlations
            std::vector<double> reals;
            std::vector<double> imags;

            for (int corr : corr_indices) {
                size_t data_idx = row_idx * n_channels * n_corr + chan_idx * n_corr + corr;
                size_t flag_idx = data_idx;

                bool is_flagged = flags_ptr[flag_idx];
                std::complex<float> value = data_ptr[data_idx];

                if (!is_flagged && std::abs(value) > 0.0) {
                    reals.push_back(std::real(value));
                    imags.push_back(std::imag(value));
                }
            }

            // Skip if no valid correlations
            if (reals.empty()) continue;

            // Calculate mean real and imaginary parts
            double real_sum = 0.0;
            double imag_sum = 0.0;
            for (size_t i = 0; i < reals.size(); ++i) {
                real_sum += reals[i];
                imag_sum += imags[i];
            }
            double real_val = real_sum / reals.size();
            double imag_val = imag_sum / imags.size();

            // Calculate mean weight
            double weight_sum = 0.0;
            for (size_t corr = 0; corr < n_corr; ++corr) {
                size_t weight_idx = row_idx * n_corr + corr;
                weight_sum += weight_ptr[weight_idx];
            }
            double w = weight_sum / n_corr;

            // Store results
            result.uv_dists.push_back(uv_dist);
            result.real_values.push_back(real_val);
            result.imag_values.push_back(imag_val);
            result.weights.push_back(w);
            result.point_indices.push_back({row_idx, (int)chan_idx});
        }
    }

    return result;
}
