#ifndef OUTLIER_DETECTION_H
#define OUTLIER_DETECTION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

/**
 * Outlier detection using MAD (Median Absolute Deviation)
 * Exact translation of Python version (navigator.py lines 379-409)
 */

struct OutlierResult {
    std::vector<bool> outliers;
    std::vector<double> residuals;
    double mad_norm;
};

OutlierResult flag_outliers_linear(
    const std::vector<double>& uv_dists,
    const std::vector<double>& amplitudes,
    const std::vector<double>& predicted_amp,
    double sigma_threshold,
    const std::vector<double>* weights = nullptr
);

// Python interface
py::dict flag_outliers(
    py::array_t<double> uv_dists,
    py::array_t<double> amplitudes,
    py::array_t<double> predicted,
    double sigma_threshold,
    py::array_t<double> weights
);

#endif // OUTLIER_DETECTION_H
