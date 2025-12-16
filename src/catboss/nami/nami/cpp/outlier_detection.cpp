#include "outlier_detection.h"
#include <algorithm>
#include <numeric>

namespace {

// Compute median
double compute_median(std::vector<double> values) {
    size_t n = values.size();
    if (n == 0) return 0.0;

    std::sort(values.begin(), values.end());

    if (n % 2 == 0) {
        return (values[n/2 - 1] + values[n/2]) / 2.0;
    } else {
        return values[n/2];
    }
}

// Compute weighted median
double compute_weighted_median(const std::vector<double>& values,
                               const std::vector<double>& weights) {
    size_t n = values.size();
    if (n == 0) return 0.0;

    // Create sorted index array
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&values](size_t a, size_t b) { return values[a] < values[b]; });

    // Compute cumulative weights
    double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    double cum_weight = 0.0;
    double half_weight = total_weight / 2.0;

    for (size_t idx : indices) {
        cum_weight += weights[idx];
        if (cum_weight >= half_weight) {
            return values[idx];
        }
    }

    return values[indices.back()];
}

} // anonymous namespace

OutlierResult flag_outliers_linear(
    const std::vector<double>& uv_dists,
    const std::vector<double>& amplitudes,
    const std::vector<double>& predicted_amp,
    double sigma_threshold,
    const std::vector<double>* weights
) {
    // Exact translation of Python code (lines 379-409):
    // predicted_amp = predict_func(uv_dists)
    // residuals = amplitudes - predicted_amp
    // ...
    // Use MAD for robust statistics
    // ...
    // outliers = np.abs(residuals - median_resid) > sigma_threshold * mad_norm

    size_t n = amplitudes.size();
    OutlierResult result;
    result.residuals.resize(n);
    result.outliers.resize(n);

    // Calculate residuals in linear space (actual - predicted)
    for (size_t i = 0; i < n; ++i) {
        result.residuals[i] = amplitudes[i] - predicted_amp[i];
    }

    double median_resid;
    double mad;

    if (weights != nullptr) {
        // Weighted median and MAD (lines 388-398)
        median_resid = compute_weighted_median(result.residuals, *weights);

        // Weighted MAD
        std::vector<double> abs_dev(n);
        for (size_t i = 0; i < n; ++i) {
            abs_dev[i] = std::abs(result.residuals[i] - median_resid);
        }
        mad = compute_weighted_median(abs_dev, *weights);

    } else {
        // Unweighted median and MAD (lines 400-401)
        median_resid = compute_median(result.residuals);

        std::vector<double> abs_dev(n);
        for (size_t i = 0; i < n; ++i) {
            abs_dev[i] = std::abs(result.residuals[i] - median_resid);
        }
        mad = compute_median(abs_dev);
    }

    // Normalize MAD (1.4826 factor makes it consistent with Gaussian std dev)
    // Line 403-404
    result.mad_norm = 1.4826 * mad;

    // Flag outliers (lines 406-407)
    for (size_t i = 0; i < n; ++i) {
        result.outliers[i] = std::abs(result.residuals[i] - median_resid) >
                            sigma_threshold * result.mad_norm;
    }

    return result;
}

py::dict flag_outliers(
    py::array_t<double> uv_dists,
    py::array_t<double> amplitudes,
    py::array_t<double> predicted,
    double sigma_threshold,
    py::array_t<double> weights_arr
) {
    auto uv_buf = uv_dists.request();
    auto amp_buf = amplitudes.request();
    auto pred_buf = predicted.request();

    size_t n = amp_buf.shape[0];

    double* amp_ptr = static_cast<double*>(amp_buf.ptr);
    double* pred_ptr = static_cast<double*>(pred_buf.ptr);

    std::vector<double> uv_vec(n);
    std::vector<double> amp_vec(n);
    std::vector<double> pred_vec(n);

    double* uv_ptr = static_cast<double*>(uv_buf.ptr);
    for (size_t i = 0; i < n; ++i) {
        uv_vec[i] = uv_ptr[i];
        amp_vec[i] = amp_ptr[i];
        pred_vec[i] = pred_ptr[i];
    }

    // Get weights if provided
    std::vector<double> weights_vec;
    std::vector<double>* weights_ptr = nullptr;

    if (weights_arr.size() > 0) {
        auto weight_buf = weights_arr.request();
        double* w_ptr = static_cast<double*>(weight_buf.ptr);
        weights_vec.resize(n);
        for (size_t i = 0; i < n; ++i) {
            weights_vec[i] = w_ptr[i];
        }
        weights_ptr = &weights_vec;
    }

    OutlierResult result = flag_outliers_linear(
        uv_vec, amp_vec, pred_vec, sigma_threshold, weights_ptr
    );

    // Convert to Python arrays
    auto outliers_arr = py::array_t<bool>(n);
    auto residuals_arr = py::array_t<double>(n);

    auto outliers_buf = outliers_arr.request();
    auto residuals_buf = residuals_arr.request();

    bool* outliers_ptr = static_cast<bool*>(outliers_buf.ptr);
    double* residuals_ptr = static_cast<double*>(residuals_buf.ptr);

    for (size_t i = 0; i < n; ++i) {
        outliers_ptr[i] = result.outliers[i];
        residuals_ptr[i] = result.residuals[i];
    }

    py::dict dict;
    dict["outliers"] = outliers_arr;
    dict["residuals"] = residuals_arr;
    dict["mad_norm"] = result.mad_norm;

    return dict;
}
