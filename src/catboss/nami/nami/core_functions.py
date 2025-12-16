"""
Core computational functions - C++ accelerated

This module provides fast C++ implementations of computational hotspots.
C++ is mandatory for NAMI performance.
"""

import numpy as np
import _nami_core


def calculate_uv_distances_fast(uvw, wavelengths):
    """
    Fast UV distance calculation (C++ implementation)

    Translates navigator.py lines 133-138
    """
    return _nami_core.calculate_uv_distances(uvw, wavelengths)


def collect_data_points_fast(data, flags, uv_distances, weight, spw_row_indices, corr_indices):
    """
    Fast data collection from visibility data (C++ implementation)

    Extracts real and imaginary parts separately (not amplitude)
    """
    # Convert to numpy arrays if needed
    spw_rows_arr = np.array(spw_row_indices, dtype=np.int32)
    corr_arr = np.array(corr_indices, dtype=np.int32)

    result = _nami_core.collect_data_points(
        data, flags, uv_distances, weight,
        spw_rows_arr, corr_arr
    )

    # Convert point_indices to list of tuples
    point_indices_arr = result['point_indices']
    point_indices = [(int(point_indices_arr[i, 0]), int(point_indices_arr[i, 1]))
                    for i in range(point_indices_arr.shape[0])]

    return {
        'uv_dists': result['uv_dists'],
        'real_values': result['real_values'],
        'imag_values': result['imag_values'],
        'weights': result['weights'],
        'point_indices': point_indices
    }


def fit_monotonic_poly_fast(uv_dists, amplitudes, weights, degree):
    """
    Fit monotonically decreasing polynomial (C++ implementation)

    Translates navigator.py lines 230-254
    """
    # C++ does the log space conversion internally
    return _nami_core.fit_monotonic_poly(uv_dists, amplitudes, weights, degree)


def flag_outliers_fast(uv_dists, amplitudes, predicted, sigma_threshold, weights=None):
    """
    Flag outliers using MAD (C++ implementation)

    Translates navigator.py lines 379-409
    """
    if weights is None:
        weights = np.array([])  # Empty array for no weights

    result = _nami_core.flag_outliers(
        uv_dists, amplitudes, predicted, sigma_threshold, weights
    )

    return result['outliers'], result['residuals'], result['mad_norm']


def fit_spline_fast(uv_dists, values, weights, n_knots=-1):
    """
    Fit cubic spline to data (C++ implementation)

    Replaces slow SLSQP polynomial optimization
    Fast weighted least squares spline fitting

    Parameters:
        uv_dists: UV distances (x-axis)
        values: real or imaginary values (y-axis)
        weights: data weights
        n_knots: number of interior knots (-1 for auto)

    Returns:
        predicted: predicted values at input UV distances
    """
    return _nami_core.fit_spline(uv_dists, values, weights, n_knots)
