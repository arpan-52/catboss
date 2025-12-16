/**
 * NAMI C++ Core Module
 *
 * Fast C++ implementations of computational hotspots
 * Exact translations of Python code from navigator.py
 *
 * NO casacore dependency - pure computation on numpy arrays
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "uv_calc.h"
#include "data_collection.h"
#include "polynomial_fit.h"
#include "outlier_detection.h"
#include "spline_fit.h"

namespace py = pybind11;

// Helper function to convert CollectedData to Python dict
py::dict collected_data_to_dict(const CollectedData& data) {
    size_t n = data.uv_dists.size();

    auto uv_arr = py::array_t<double>(n);
    auto real_arr = py::array_t<double>(n);
    auto imag_arr = py::array_t<double>(n);
    auto weight_arr = py::array_t<double>(n);

    // Create 2D array with correct pybind11 syntax
    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(n), 2};
    auto indices_arr = py::array_t<int>(shape);

    auto uv_buf = uv_arr.request();
    auto real_buf = real_arr.request();
    auto imag_buf = imag_arr.request();
    auto weight_buf = weight_arr.request();
    auto indices_buf = indices_arr.request();

    double* uv_ptr = static_cast<double*>(uv_buf.ptr);
    double* real_ptr = static_cast<double*>(real_buf.ptr);
    double* imag_ptr = static_cast<double*>(imag_buf.ptr);
    double* weight_ptr = static_cast<double*>(weight_buf.ptr);
    int* indices_ptr = static_cast<int*>(indices_buf.ptr);

    for (size_t i = 0; i < n; ++i) {
        uv_ptr[i] = data.uv_dists[i];
        real_ptr[i] = data.real_values[i];
        imag_ptr[i] = data.imag_values[i];
        weight_ptr[i] = data.weights[i];
        indices_ptr[i * 2 + 0] = data.point_indices[i].first;
        indices_ptr[i * 2 + 1] = data.point_indices[i].second;
    }

    py::dict result;
    result["uv_dists"] = uv_arr;
    result["real_values"] = real_arr;
    result["imag_values"] = imag_arr;
    result["weights"] = weight_arr;
    result["point_indices"] = indices_arr;

    return result;
}

// Wrapper for collect_data_points to match Python API
py::dict collect_data_points_py(
    py::array_t<std::complex<float>> data,
    py::array_t<bool> flags,
    py::array_t<double> uv_distances,
    py::array_t<float> weight,
    py::array_t<int> spw_row_indices,
    py::array_t<int> corr_indices
) {
    auto spw_buf = spw_row_indices.request();
    auto corr_buf = corr_indices.request();

    int* spw_ptr = static_cast<int*>(spw_buf.ptr);
    int* corr_ptr = static_cast<int*>(corr_buf.ptr);

    std::vector<int> spw_vec(spw_ptr, spw_ptr + spw_buf.shape[0]);
    std::vector<int> corr_vec(corr_ptr, corr_ptr + corr_buf.shape[0]);

    CollectedData result = collect_data_points(
        data, flags, uv_distances, weight, spw_vec, corr_vec
    );

    return collected_data_to_dict(result);
}

PYBIND11_MODULE(_nami_core, m) {
    m.doc() = "NAMI C++ Core - Fast implementations of computational hotspots";

    // UV distance calculation
    m.def("calculate_uv_distances", &calculate_uv_distances,
          py::arg("uvw"),
          py::arg("wavelengths"),
          R"pbdoc(
            Fast UV distance calculation

            Exact C++ translation of navigator.py lines 133-138

            Parameters:
                uvw: numpy array shape (n_rows, 3) - UVW coordinates
                wavelengths: numpy array shape (n_channels,) - wavelengths in meters

            Returns:
                numpy array shape (n_rows, n_channels) - UV distances in wavelengths
          )pbdoc");

    // Data collection
    m.def("collect_data_points", &collect_data_points_py,
          py::arg("data"),
          py::arg("flags"),
          py::arg("uv_distances"),
          py::arg("weight"),
          py::arg("spw_row_indices"),
          py::arg("corr_indices"),
          R"pbdoc(
            Fast data collection from visibility data

            Extracts real and imaginary parts separately (not amplitude)

            Parameters:
                data: complex array shape (n_rows, n_channels, n_corr)
                flags: bool array shape (n_rows, n_channels, n_corr)
                uv_distances: float array shape (n_rows, n_channels)
                weight: float array shape (n_rows, n_corr)
                spw_row_indices: int array - which rows to process
                corr_indices: int array - which correlations to use

            Returns:
                dict with keys: uv_dists, real_values, imag_values, weights, point_indices
          )pbdoc");

    // Polynomial fitting
    m.def("fit_monotonic_poly", &fit_monotonic_poly,
          py::arg("uv_dists"),
          py::arg("amplitudes"),
          py::arg("weights"),
          py::arg("degree"),
          R"pbdoc(
            Fit monotonically decreasing polynomial in log space

            Exact C++ translation of navigator.py lines 230-254

            Parameters:
                uv_dists: UV distances
                amplitudes: visibility amplitudes
                weights: data weights
                degree: polynomial degree

            Returns:
                numpy array - polynomial coefficients (for use with np.polyval)
          )pbdoc");

    // Outlier detection
    m.def("flag_outliers", &flag_outliers,
          py::arg("uv_dists"),
          py::arg("amplitudes"),
          py::arg("predicted"),
          py::arg("sigma_threshold"),
          py::arg("weights"),
          R"pbdoc(
            Flag outliers using MAD (Median Absolute Deviation)

            Exact C++ translation of navigator.py lines 379-409

            Parameters:
                uv_dists: UV distances
                amplitudes: actual amplitudes
                predicted: predicted amplitudes from model
                sigma_threshold: flagging threshold in sigmas
                weights: data weights (optional)

            Returns:
                dict with keys: outliers (bool array), residuals, mad_norm
          )pbdoc");

    // Spline fitting (replaces slow polynomial optimization)
    m.def("fit_spline", &fit_spline,
          py::arg("uv_dists"),
          py::arg("values"),
          py::arg("weights"),
          py::arg("n_knots") = -1,
          R"pbdoc(
            Fit cubic spline to data (real or imaginary part)

            Fast spline fitting using weighted least squares
            Replaces slow SLSQP polynomial optimization

            Parameters:
                uv_dists: UV distances (x-axis)
                values: real or imaginary values (y-axis)
                weights: data weights
                n_knots: number of interior knots (-1 for auto)

            Returns:
                numpy array - predicted values at input UV distances
          )pbdoc");

    m.attr("__version__") = "1.0.0";
}
