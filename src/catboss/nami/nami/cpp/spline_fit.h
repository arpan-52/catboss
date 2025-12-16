/**
 * Cubic Spline Fitting for NAMI
 *
 * Fast cubic spline interpolation/smoothing for UV-distance vs visibility data
 * Replaces slow SLSQP polynomial optimization with direct linear solver
 *
 * Expected speedup: 60s -> <0.1s per fit
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

/**
 * Cubic spline fitter with automatic knot placement
 *
 * Uses natural cubic splines with weighted least squares fitting
 * Knots are placed adaptively based on data density in UV space
 */
class CubicSplineFitter {
public:
    /**
     * Fit a cubic spline to data
     *
     * @param uv_dists: UV distances (x-axis)
     * @param values: Real or imaginary values (y-axis)
     * @param weights: Data weights
     * @param n_knots: Number of interior knots (default: auto-select)
     */
    void fit(const std::vector<double>& uv_dists,
             const std::vector<double>& values,
             const std::vector<double>& weights,
             int n_knots = -1);

    /**
     * Evaluate fitted spline at given points
     *
     * @param uv_points: Points to evaluate at
     * @return: Predicted values
     */
    std::vector<double> evaluate(const std::vector<double>& uv_points) const;

    /**
     * Get spline coefficients (for serialization/debugging)
     */
    const std::vector<double>& get_coefficients() const { return coeffs_; }
    const std::vector<double>& get_knots() const { return knots_; }

private:
    std::vector<double> knots_;      // Knot positions
    std::vector<double> coeffs_;     // Spline coefficients

    /**
     * Place knots adaptively based on data distribution
     */
    void place_knots(const std::vector<double>& uv_dists, int n_knots);

    /**
     * Build and solve linear system for spline coefficients
     */
    void solve_spline_system(const std::vector<double>& uv_dists,
                            const std::vector<double>& values,
                            const std::vector<double>& weights);

    /**
     * Evaluate cubic B-spline basis function
     */
    double basis_function(double x, int i) const;
};

/**
 * Python-facing function: Fit spline and return predicted values
 *
 * @param uv_dists: numpy array of UV distances
 * @param values: numpy array of real or imaginary values
 * @param weights: numpy array of weights
 * @param n_knots: number of knots (or -1 for auto)
 * @return: numpy array of predicted values at input UV distances
 */
py::array_t<double> fit_spline(
    py::array_t<double> uv_dists,
    py::array_t<double> values,
    py::array_t<double> weights,
    int n_knots = -1
);
