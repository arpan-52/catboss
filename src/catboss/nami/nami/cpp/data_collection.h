#ifndef DATA_COLLECTION_H
#define DATA_COLLECTION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>
#include <vector>
#include <cmath>

namespace py = pybind11;

/**
 * Fast data collection - extract real and imaginary parts separately
 * Modified to fit real and imag independently (not amplitude)
 */
struct CollectedData {
    std::vector<double> uv_dists;
    std::vector<double> real_values;      // Real part of visibility
    std::vector<double> imag_values;      // Imaginary part of visibility
    std::vector<double> weights;
    std::vector<std::pair<int, int>> point_indices; // (row_idx, chan_idx)
};

CollectedData collect_data_points(
    py::array_t<std::complex<float>> data,  // Shape: (n_rows, n_channels, n_corr)
    py::array_t<bool> flags,                // Shape: (n_rows, n_channels, n_corr)
    py::array_t<double> uv_distances,       // Shape: (n_rows, n_channels)
    py::array_t<float> weight,              // Shape: (n_rows, n_corr)
    std::vector<int> spw_row_indices,       // Which rows belong to this SPW
    std::vector<int> corr_indices           // Which correlations to use
);

#endif // DATA_COLLECTION_H
