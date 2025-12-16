#ifndef UV_CALC_H
#define UV_CALC_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

/**
 * Fast UV distance calculation - exact translation of Python version
 * Python code (navigator.py lines 133-138):
 *   for i, wavelength in enumerate(wavelengths[spw]):
 *       uv_dist_per_channel[spw][:, i] = np.sqrt(uvw[:, 0]**2 + uvw[:, 1]**2) / wavelength
 */
py::array_t<double> calculate_uv_distances(
    py::array_t<double> uvw,        // Shape: (n_rows, 3)
    py::array_t<double> wavelengths // Shape: (n_channels,)
);

#endif // UV_CALC_H
