#include <stdio.h>
#include <chrono>
#include <stdio.h>
#include <string.h>
#include <string>
#include <ctime>
#include <cstdlib>
#include <math.h>

#include "common.h"
#include "utils.h"


double ponn_utils_get_span(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
}

bool ponn_utils_cmp(const std::pair<std::string,float> a, std::pair<std::string,float>b){
    return a.second > b.second;
}

void ponn_utils_tensor_precision_info(ggml_tensor* tensor) {
    size_t size = 0;
    void *data = nullptr;
    float *f32_data = nullptr;
    double mean = 0;
    double std = 0;
    size = ggml_nbytes(tensor);
    PONN_DATA_TYPE_E  pDataType =  ponnGetInferenceDataType();
    if (tensor->extra) {
        data = malloc(size);
        ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)tensor->extra;
        if(tensor->type == GGML_TYPE_F32 && pDataType == PONN_DATA_HALF) {
            std::vector<int> tensor_dims;
            ponn_utils_get_tensor_dims(tensor_dims, tensor->ne);
            ponnMemcpyEx(data, ponn_utils_get_data_type(tensor->type),  extra->handle, pDataType,  size, tensor_dims, DEVICE_TO_HOST);
        }else {
            ponnMemcpy(data, 0, extra->handle, extra->offset,  size, DEVICE_TO_HOST);
        }
        ponnSyncStream(ponnGetStream());
    } else {
        data = tensor->data;
    }
    if (tensor->type == GGML_TYPE_F32) {
        f32_data = (float *)data;
    } else if (tensor->type == GGML_TYPE_F16) {
        f32_data = (float *)malloc(size * 2);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)data, f32_data, size / 2);
    }

    if (f32_data) {
        mean = compute_mean<float>((float *)f32_data, size / 4);
        std = compute_standard_deviation<float>((float *)f32_data, size / 4);
        GGML_LOG_INFO("mean %10.5f  std %10.5f", mean, std);
    }
    if (f32_data && f32_data != data) {
        free(f32_data);
    }
    if (data && data != tensor->data) {
        free(data);
    }
}

bool ponn_utils_is_padded_1d_0(const struct ggml_tensor * tensor) {
    return
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[1] > (tensor->nb[0]*tensor->ne[0]) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}


void ponn_utils_get_stride_div(ggml_type gDataType, size_t &stride_div) {
    PONN_DATA_TYPE_E  pDataType =  ponnGetInferenceDataType();
    if(pDataType == PONN_DATA_HALF &&
        gDataType == GGML_TYPE_F32) {
            stride_div = 2;
        }
}

PONN_DATA_TYPE_E ponn_utils_get_data_type(const ggml_type dtype){
    switch (dtype)
    {
    case GGML_TYPE_F32:
        return PONN_DATA_FLOAT;
    case GGML_TYPE_F16:
        return PONN_DATA_HALF;
    case GGML_TYPE_Q4_1:
        return PONN_DATA_LLAMA_Q4_1;
    case GGML_TYPE_Q4_K:
        return PONN_DATA_LLAMA_Q4_K;
    default:
    //  GGML_ASSERT(false);
        break;
    }
    return PONN_DATA_FLOAT;
}

void ponn_utils_get_tensor_dims(std::vector<int>& dims, const int64_t* ne){
    dims.clear();
    for(int i = 1; i < GGML_MAX_DIMS + 1; ++i) {
        dims.push_back(static_cast<int>(ne[GGML_MAX_DIMS - i]));
    }
}
