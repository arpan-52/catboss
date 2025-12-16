#include "uv_calc.h"

py::array_t<double> calculate_uv_distances(
    py::array_t<double> uvw,
    py::array_t<double> wavelengths
) {
    // Get array info
    auto uvw_buf = uvw.request();
    auto wave_buf = wavelengths.request();

    if (uvw_buf.ndim != 2 || uvw_buf.shape[1] != 3) {
        throw std::runtime_error("UVW must be shape (n_rows, 3)");
    }
    if (wave_buf.ndim != 1) {
        throw std::runtime_error("Wavelengths must be 1D array");
    }

    size_t n_rows = uvw_buf.shape[0];
    size_t n_channels = wave_buf.shape[0];

    // Access data
    double* uvw_ptr = static_cast<double*>(uvw_buf.ptr);
    double* wave_ptr = static_cast<double*>(wave_buf.ptr);

    // Create output array
    auto result = py::array_t<double>({n_rows, n_channels});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    // Exact translation of Python loop:
    // for i, wavelength in enumerate(wavelengths[spw]):
    //     uv_dist_per_channel[spw][:, i] = np.sqrt(uvw[:, 0]**2 + uvw[:, 1]**2) / wavelength

    #pragma omp parallel for
    for (size_t row = 0; row < n_rows; ++row) {
        double u = uvw_ptr[row * 3 + 0];
        double v = uvw_ptr[row * 3 + 1];
        double uv_magnitude = std::sqrt(u * u + v * v);

        for (size_t chan = 0; chan < n_channels; ++chan) {
            double wavelength = wave_ptr[chan];
            result_ptr[row * n_channels + chan] = uv_magnitude / wavelength;
        }
    }

    return result;
}
