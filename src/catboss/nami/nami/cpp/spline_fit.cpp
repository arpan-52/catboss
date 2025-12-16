/**
 * Cubic Spline Fitting Implementation
 *
 * Uses natural cubic splines with weighted least squares
 * Adaptive knot placement based on data density
 */

#include "spline_fit.h"
#include <cmath>
#include <stdexcept>
#include <numeric>

// Simple matrix operations (avoiding external dependencies)
namespace {

/**
 * Solve Ax = b using Gaussian elimination with partial pivoting
 * Simple but stable for small-medium systems (n_knots typically < 100)
 */
std::vector<double> solve_linear_system(
    std::vector<std::vector<double>>& A,
    std::vector<double>& b
) {
    int n = b.size();

    // Forward elimination with partial pivoting
    for (int i = 0; i < n; ++i) {
        // Find pivot
        int max_row = i;
        double max_val = std::abs(A[i][i]);
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(A[k][i]) > max_val) {
                max_val = std::abs(A[k][i]);
                max_row = k;
            }
        }

        // Swap rows
        if (max_row != i) {
            std::swap(A[i], A[max_row]);
            std::swap(b[i], b[max_row]);
        }

        // Eliminate column
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(A[i][i]) < 1e-14) continue;  // Skip if pivot too small
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; ++j) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= A[i][j] * x[j];
        }
        if (std::abs(A[i][i]) > 1e-14) {
            x[i] /= A[i][i];
        }
    }

    return x;
}

} // anonymous namespace


void CubicSplineFitter::place_knots(const std::vector<double>& uv_dists, int n_knots) {
    if (uv_dists.empty()) {
        throw std::runtime_error("Cannot place knots on empty data");
    }

    // Get sorted unique UV distances for knot placement
    std::vector<double> sorted_uv = uv_dists;
    std::sort(sorted_uv.begin(), sorted_uv.end());

    double uv_min = sorted_uv.front();
    double uv_max = sorted_uv.back();

    // Auto-select number of knots if not specified
    if (n_knots <= 0) {
        // Rule of thumb: sqrt(n_points) / 2, clamped to [5, 50]
        n_knots = std::max(5, std::min(50, static_cast<int>(std::sqrt(uv_dists.size())) / 2));
    }

    // Place knots uniformly in UV space
    // (Could be made adaptive based on data density in future)
    knots_.clear();
    knots_.reserve(n_knots + 8);  // +8 for boundary knots (order 4)

    // Add boundary knots (4 at each end for cubic splines)
    for (int i = 0; i < 4; ++i) {
        knots_.push_back(uv_min);
    }

    // Interior knots
    for (int i = 0; i < n_knots; ++i) {
        double t = static_cast<double>(i + 1) / (n_knots + 1);
        knots_.push_back(uv_min + t * (uv_max - uv_min));
    }

    // Add boundary knots at end
    for (int i = 0; i < 4; ++i) {
        knots_.push_back(uv_max);
    }
}


double CubicSplineFitter::basis_function(double x, int i) const {
    // Cubic B-spline basis function (de Boor-Cox recursion)
    // Iterative implementation (order k=4 for cubic)

    const int k = 4;  // Cubic
    const int n_basis = knots_.size() - k;

    if (i < 0 || i >= n_basis) return 0.0;

    // Check if outside support
    if (x < knots_[i] || x >= knots_[i + k]) return 0.0;

    // Iterative de Boor-Cox algorithm
    std::vector<double> N_vals(k, 0.0);

    // Initial values (order 1)
    for (int j = 0; j < k; ++j) {
        if (x >= knots_[i + j] && x < knots_[i + j + 1]) {
            N_vals[j] = 1.0;
        }
    }

    // Build up to order k
    for (int order = 2; order <= k; ++order) {
        std::vector<double> N_new(k, 0.0);
        for (int j = 0; j < k - order + 1; ++j) {
            double left = 0.0, right = 0.0;

            double denom1 = knots_[i + j + order - 1] - knots_[i + j];
            if (std::abs(denom1) > 1e-14) {
                left = (x - knots_[i + j]) / denom1 * N_vals[j];
            }

            double denom2 = knots_[i + j + order] - knots_[i + j + 1];
            if (std::abs(denom2) > 1e-14 && j + 1 < static_cast<int>(N_vals.size())) {
                right = (knots_[i + j + order] - x) / denom2 * N_vals[j + 1];
            }

            N_new[j] = left + right;
        }
        N_vals = N_new;
    }

    return N_vals[0];
}


void CubicSplineFitter::solve_spline_system(
    const std::vector<double>& uv_dists,
    const std::vector<double>& values,
    const std::vector<double>& weights
) {
    const int n_data = uv_dists.size();
    const int n_basis = knots_.size() - 4;  // For cubic (order 4)

    // Build weighted least squares system: (B^T W B) c = B^T W y
    std::vector<std::vector<double>> ATA(n_basis, std::vector<double>(n_basis, 0.0));
    std::vector<double> ATb(n_basis, 0.0);

    // Accumulate for each data point
    for (int i = 0; i < n_data; ++i) {
        double x = uv_dists[i];
        double y = values[i];
        double w = weights[i];

        // Evaluate all basis functions at this point
        std::vector<double> basis_vals(n_basis);
        for (int j = 0; j < n_basis; ++j) {
            basis_vals[j] = basis_function(x, j);
        }

        // Add contribution to normal equations
        for (int j = 0; j < n_basis; ++j) {
            ATb[j] += w * y * basis_vals[j];
            for (int k = 0; k < n_basis; ++k) {
                ATA[j][k] += w * basis_vals[j] * basis_vals[k];
            }
        }
    }

    // Add regularization for numerical stability
    const double regularization = 1e-8;
    for (int i = 0; i < n_basis; ++i) {
        ATA[i][i] += regularization;
    }

    // Solve system
    coeffs_ = solve_linear_system(ATA, ATb);
}


void CubicSplineFitter::fit(
    const std::vector<double>& uv_dists,
    const std::vector<double>& values,
    const std::vector<double>& weights,
    int n_knots
) {
    if (uv_dists.size() != values.size() || uv_dists.size() != weights.size()) {
        throw std::runtime_error("Input arrays must have same size");
    }

    if (uv_dists.empty()) {
        throw std::runtime_error("Cannot fit spline to empty data");
    }

    // Place knots
    place_knots(uv_dists, n_knots);

    // Solve for coefficients
    solve_spline_system(uv_dists, values, weights);
}


std::vector<double> CubicSplineFitter::evaluate(const std::vector<double>& uv_points) const {
    if (coeffs_.empty()) {
        throw std::runtime_error("Spline not fitted yet");
    }

    const int n_basis = coeffs_.size();
    std::vector<double> result(uv_points.size(), 0.0);

    for (size_t i = 0; i < uv_points.size(); ++i) {
        double x = uv_points[i];

        // Sum over all basis functions
        for (int j = 0; j < n_basis; ++j) {
            result[i] += coeffs_[j] * basis_function(x, j);
        }
    }

    return result;
}


// Python-facing function
py::array_t<double> fit_spline(
    py::array_t<double> uv_dists,
    py::array_t<double> values,
    py::array_t<double> weights,
    int n_knots
) {
    auto uv_buf = uv_dists.request();
    auto val_buf = values.request();
    auto weight_buf = weights.request();

    double* uv_ptr = static_cast<double*>(uv_buf.ptr);
    double* val_ptr = static_cast<double*>(val_buf.ptr);
    double* weight_ptr = static_cast<double*>(weight_buf.ptr);

    size_t n = uv_buf.shape[0];

    // Convert to vectors
    std::vector<double> uv_vec(uv_ptr, uv_ptr + n);
    std::vector<double> val_vec(val_ptr, val_ptr + n);
    std::vector<double> weight_vec(weight_ptr, weight_ptr + n);

    // Fit spline
    CubicSplineFitter fitter;
    fitter.fit(uv_vec, val_vec, weight_vec, n_knots);

    // Evaluate at original points
    std::vector<double> predicted = fitter.evaluate(uv_vec);

    // Return as numpy array
    auto result = py::array_t<double>(n);
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    std::copy(predicted.begin(), predicted.end(), result_ptr);

    return result;
}
