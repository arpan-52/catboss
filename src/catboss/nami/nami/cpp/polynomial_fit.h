#ifndef POLYNOMIAL_FIT_H
#define POLYNOMIAL_FIT_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

/**
 * Polynomial fitting with monotonic constraints
 * Exact translation of Python version (navigator.py lines 189-254)
 */

// Fit polynomial with optional weights (unconstrained)
std::vector<double> fit_polynomial_unconstrained(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int degree,
    const std::vector<double>* weights = nullptr
);

// Check if polynomial is monotonically decreasing
bool is_monotonic_decreasing(
    const std::vector<double>& coef,
    double x_min,
    double x_max,
    int num_test_points = 50
);

// Fit polynomial with monotonicity constraint
std::vector<double> fit_constrained_polynomial(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int degree,
    const std::vector<double>* weights = nullptr
);

// Main interface: fit monotonic polynomial in log space
// Returns coefficients that can be used with np.polyval
py::array_t<double> fit_monotonic_poly(
    py::array_t<double> uv_dists,
    py::array_t<double> amplitudes,
    py::array_t<double> weights,
    int degree
);

#endif // POLYNOMIAL_FIT_H
