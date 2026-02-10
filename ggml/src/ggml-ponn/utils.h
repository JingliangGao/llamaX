#ifndef PONN_UTILS_H
#define PONN_UTILS_H
#include <math.h>
#include "ggml.h"

#ifdef  __cplusplus

template <typename T>
double compute_mean(const T* in, const size_t length) {
    double sum = 0.;
    for (size_t i = 0; i < length; ++i) {
      sum += in[i];
    }
    return sum / length;
}

template <typename T>
double compute_standard_deviation(const T* in,
                                    const size_t length,
                                    bool has_mean = false,
                                    double mean = 10000) {
    if (!has_mean) {
        mean = compute_mean<T>(in, length);
    }

    double variance = 0.;
    for (size_t i = 0; i < length; ++i) {
        variance += pow((in[i] - mean), 2);
    }
    variance /= length;
    return sqrt(variance);
}

bool ponn_utils_cmp(const std::pair<std::string, float> a, std::pair<std::string, float> b);
double ponn_utils_get_span(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2);

bool ponn_utils_is_padded_1d_0(const struct ggml_tensor * tensor);

PONN_DATA_TYPE_E ponn_utils_get_data_type(const ggml_type dtype);
void ponn_utils_get_stride_div(ggml_type g_type, size_t &stride_div);
void ponn_utils_get_tensor_dims(std::vector<int>& dims, const int64_t* ne);
#endif

#ifdef  __cplusplus
extern "C" {
#endif
void ponn_utils_tensor_precision_info(struct ggml_tensor* tensor);

#ifdef  __cplusplus
}
#endif

#endif  // PONN_UTILS_H
