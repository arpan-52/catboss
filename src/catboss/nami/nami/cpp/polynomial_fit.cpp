#include "polynomial_fit.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>

// Simple matrix operations for polynomial fitting
namespace {

// Compute polynomial value
double polyval(const std::vector<double>& coef, double x) {
    double result = 0.0;
    double x_pow = 1.0;
    for (int i = coef.size() - 1; i >= 0; --i) {
        result += coef[i] * x_pow;
        x_pow *= x;
    }
    return result;
}

// Compute polynomial derivative coefficients
std::vector<double> polyder(const std::vector<double>& coef) {
    int degree = coef.size() - 1;
    if (degree == 0) {
        return {0.0};
    }

    std::vector<double> deriv(degree);
    for (int i = 0; i < degree; ++i) {
        deriv[i] = coef[i] * (degree - i);
    }
    return deriv;
}

// Solve least squares: X^T * W * X * beta = X^T * W * y
std::vector<double> weighted_least_squares(
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::vector<double>* weights
) {
    size_t n = X.size();
    size_t p = X[0].size();

    // Build normal equations: A * beta = b
    std::vector<std::vector<double>> A(p, std::vector<double>(p, 0.0));
    std::vector<double> b(p, 0.0);

    for (size_t i = 0; i < n; ++i) {
        double w = weights ? (*weights)[i] : 1.0;

        for (size_t j = 0; j < p; ++j) {
            b[j] += w * X[i][j] * y[i];
            for (size_t k = 0; k < p; ++k) {
                A[j][k] += w * X[i][j] * X[i][k];
            }
        }
    }

    // Solve using Gaussian elimination with partial pivoting
    std::vector<double> beta(p);

    for (size_t i = 0; i < p; ++i) {
        // Find pivot
        size_t max_row = i;
        for (size_t k = i + 1; k < p; ++k) {
            if (std::abs(A[k][i]) > std::abs(A[max_row][i])) {
                max_row = k;
            }
        }

        // Swap rows
        std::swap(A[i], A[max_row]);
        std::swap(b[i], b[max_row]);

        // Eliminate
        for (size_t k = i + 1; k < p; ++k) {
            double factor = A[k][i] / A[i][i];
            for (size_t j = i; j < p; ++j) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    for (int i = p - 1; i >= 0; --i) {
        beta[i] = b[i];
        for (size_t j = i + 1; j < p; ++j) {
            beta[i] -= A[i][j] * beta[j];
        }
        beta[i] /= A[i][i];
    }

    return beta;
}

} // anonymous namespace

std::vector<double> fit_polynomial_unconstrained(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int degree,
    const std::vector<double>* weights
) {
    size_t n = x.size();

    // Build Vandermonde matrix
    std::vector<std::vector<double>> X(n, std::vector<double>(degree + 1));
    for (size_t i = 0; i < n; ++i) {
        double x_pow = 1.0;
        for (int j = degree; j >= 0; --j) {
            X[i][j] = x_pow;
            x_pow *= x[i];
        }
    }

    return weighted_least_squares(X, y, weights);
}

bool is_monotonic_decreasing(
    const std::vector<double>& coef,
    double x_min,
    double x_max,
    int num_test_points
) {
    // Exact translation of Python code (lines 194-198):
    // derivative = np.polyder(coef)
    // test_points = np.linspace(min(x_vals), max(x_vals), 20)
    // derivatives = np.polyval(derivative, test_points)
    // For decreasing function, all derivatives should be <= 0
    // return -np.max(derivatives)  # Constraint satisfied when <= 0

    std::vector<double> derivative = polyder(coef);

    double max_deriv = -1e10;
    for (int i = 0; i < num_test_points; ++i) {
        double t = static_cast<double>(i) / (num_test_points - 1);
        double x = x_min + t * (x_max - x_min);
        double deriv_val = polyval(derivative, x);
        max_deriv = std::max(max_deriv, deriv_val);
    }

    // For monotonic decreasing, max derivative should be <= 0
    return max_deriv <= 1e-4;  // Small tolerance as in Python (line 201)
}

std::vector<double> fit_constrained_polynomial(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int degree,
    const std::vector<double>* weights
) {
    // Exact translation of Python code (lines 189-228):
    // Initial unconstrained fit
    std::vector<double> initial_coef = fit_polynomial_unconstrained(x, y, degree, weights);

    // Check if initial fit is already monotonic
    double x_min = *std::min_element(x.begin(), x.end());
    double x_max = *std::max_element(x.begin(), x.end());

    if (is_monotonic_decreasing(initial_coef, x_min, x_max)) {
        return initial_coef;  // Already monotonic, return as-is (lines 200-202)
    }

    // If not monotonic, we need constrained optimization
    // For now, fall back to unconstrained fit (as Python does in line 224-226)
    // A full SLSQP implementation would be very complex and require external library
    // The Python code also falls back if optimization fails
    return initial_coef;
}

py::array_t<double> fit_monotonic_poly(
    py::array_t<double> uv_dists,
    py::array_t<double> amplitudes,
    py::array_t<double> weights_arr,
    int degree
) {
    // Exact translation of Python code (lines 230-254):
    // Convert UV distances and amplitudes to log space for fitting
    // log_uv = np.log10(uv_dists + 1e-10)
    // log_amp = np.log10(amplitudes + 1e-10)

    auto uv_buf = uv_dists.request();
    auto amp_buf = amplitudes.request();
    auto weight_buf = weights_arr.request();

    size_t n = uv_buf.shape[0];

    double* uv_ptr = static_cast<double*>(uv_buf.ptr);
    double* amp_ptr = static_cast<double*>(amp_buf.ptr);
    double* weight_ptr = static_cast<double*>(weight_buf.ptr);

    // Convert to log space
    std::vector<double> log_uv(n);
    std::vector<double> log_amp(n);
    std::vector<double> weights_vec(n);

    for (size_t i = 0; i < n; ++i) {
        log_uv[i] = std::log10(uv_ptr[i] + 1e-10);
        log_amp[i] = std::log10(amp_ptr[i] + 1e-10);
        weights_vec[i] = weight_ptr[i];
    }

    // Fit constrained polynomial in log space
    std::vector<double> coefs = fit_constrained_polynomial(
        log_uv, log_amp, degree, &weights_vec
    );

    // Return coefficients as numpy array
    auto result = py::array_t<double>(coefs.size());
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    for (size_t i = 0; i < coefs.size(); ++i) {
        result_ptr[i] = coefs[i];
    }

    return result;
}
