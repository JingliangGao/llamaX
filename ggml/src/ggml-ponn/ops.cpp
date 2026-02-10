#include "ops.h"

#include <float.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "utils.h"
#include "ponn.h"
#include "ggml.h"
#include "common.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"

PONN_MEM_H ponnPrepare(const ggml_tensor* tensor) {
    GGML_ASSERT(tensor->extra);
    void *ret;
    ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)tensor->extra;
    size_t size = ggml_nbytes(tensor);
    GGML_ASSERT(!extra->offset);
    //GGML_ASSERT(!tensor->view_offs);
    return extra->handle;
}

void ponnFinish(const ggml_tensor* tensor, PONN_MEM_H data) {

}

#ifdef GGML_PONN_CHECK
void ggml_ponn_check(ggml_tensor *dst, float align) {
    ggml_tensor_extra_gpu *extrad = (ggml_tensor_extra_gpu *)dst->extra;
    void *gpu_res_buffer = malloc(ggml_nbytes(dst));
    ponnMemcpy(gpu_res_buffer, 0, extrad->handle, extrad->offset, ggml_nbytes(dst), DEVICE_TO_HOST);
    ggml_ponn_fallback(dst);
    void *cpu_res_buffer = malloc(ggml_nbytes(dst));
    ponnMemcpy(cpu_res_buffer, 0, extrad->handle, extrad->offset, ggml_nbytes(dst), DEVICE_TO_HOST);
    if(ggml_is_contiguous(dst)) {
        for(int i=0; i<ggml_nelements(dst); ++i) {
            float cpu_res, gpu_res;
            if(dst->type == GGML_TYPE_F16) {
                cpu_res = ggml_fp16_to_fp32(*((ggml_fp16_t *)cpu_res_buffer + i));
                gpu_res = ggml_fp16_to_fp32(*((ggml_fp16_t *)gpu_res_buffer + i));
            } else {
                cpu_res = *((float *)cpu_res_buffer + i);
                gpu_res = *((float *)gpu_res_buffer + i);
            }
            GGML_ASSERT(abs(cpu_res - gpu_res) <= align);
        }
    } else {
        for(int i3=0; i3<dst->ne[3]; i3++) {
            for(int i2=0; i2<dst->ne[2]; i2++) {
                for(int i1=0; i1<dst->ne[1]; i1++) {
                    for(int i0=0; i0<dst->ne[0]; i0++) {
                        char *gpu_ptr = (char *)gpu_res_buffer + i0*dst->nb[0] + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];
                        char *cpu_ptr = (char *)cpu_res_buffer + i0*dst->nb[0] + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];
                        float cpu_res, gpu_res;
                        if(dst->type == GGML_TYPE_F16) {
                            gpu_res = ggml_fp16_to_fp32(*(ggml_fp16_t *)gpu_ptr);
                            cpu_res = ggml_fp16_to_fp32(*(ggml_fp16_t *)cpu_ptr);
                        } else {
                            gpu_res = *(float *)gpu_ptr;
                            cpu_res = *(float *)cpu_ptr;
                        }
                        GGML_ASSERT(abs(cpu_res - gpu_res) <= align);
                    }
                }
            }
        }
    }
    free(gpu_res_buffer);
    free(cpu_res_buffer);
}
#endif

void ggml_ponn_repeat(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(ggml_can_repeat(src, dst));
    GGML_ASSERT(0);
}


void ggml_ponn_add(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    PONN_MEM_H input0 = ponnPrepare(src0);
    ggml_tensor_extra_gpu* input0_extra =(ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu* dst_extra =(ggml_tensor_extra_gpu *) dst->extra;
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    std::vector<int> input1Dims = {src1->ne[3], src1->ne[2], src1->ne[1], src1->ne[0]};
    std::vector<int> outputDims = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
    ponnAdd(input0, input1, output, input0Dims, input1Dims, outputDims);
    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
    ponnFinish(dst, output);
    return;
}

void ggml_ponn_leaky_relu(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(0);
}

void ggml_ponn_concat(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(0);
}

void ggml_ponn_arange(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int64_t n_elements = ggml_nelements(dst);
    float start;
    float stop;
    float step;
    memcpy(&start, (float*)dst->op_params + 0, sizeof(float));
    memcpy(&stop, (float*)dst->op_params + 1, sizeof(float));
    memcpy(&step, (float*)dst->op_params + 2, sizeof(float));
    GGML_ASSERT(0);
}

void ggml_ponn_sqr(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    dst->src[1] = dst->src[0];
    GGML_ASSERT(0);
}

void ggml_ponn_clamp(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float*)dst->op_params + 1, sizeof(float));
    GGML_ASSERT(0);
}

void ggml_ponn_scale(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    float v;
    memcpy(&v, dst->op_params, sizeof(float));

    PONN_MEM_H input = ponnPrepare(src0);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    std::vector<int> outputDims = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
    ponnScale(input, output, input0Dims, v);

    ponnFinish(src0, input);
    ponnFinish(dst, output);
    return;
}

void ggml_ponn_argsort(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    enum ggml_sort_order order = (enum ggml_sort_order)dst->op_params[0];
    GGML_ASSERT(0);
}

void ggml_ponn_norm(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(0);
}

void ggml_ponn_group_norm(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    int n_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(0);
}

void ggml_ponn_acc(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    size_t nb1 = ((int32_t*)dst->op_params)[0];
    size_t nb2 = ((int32_t*)dst->op_params)[1];
    size_t nb3 = ((int32_t*)dst->op_params)[2];
    size_t offset = ((int32_t*)dst->op_params)[3];
    bool inplace = (bool)((int32_t*)dst->op_params)[4];
    GGML_ASSERT(0);
}

void ggml_ponn_sum_rows(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(0);
}

void ggml_ponn_upsample_nearest2d(ggml_backend_ponn_context& ctx,
                                  ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(0);
}

void ggml_ponn_pad(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(0);
}

/**
 * @brief Performs 2D average pooling on the input tensor and stores the result
 * in the destination tensor.
 *
 * This function performs average pooling on the source tensor and stores the
 * result in the destination tensor. The pooling parameters (kernel size,
 * strides, padding) are specified in the `op_params` of the destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result will be stored. The source
 * tensor is referenced by `dst->src[0]`.
 */
static void ggml_ponn_avg_pool2d(ggml_backend_ponn_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    const int32_t* opts = (const int32_t*)dst->op_params;
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];
    GGML_ASSERT(0);
}

/**
 * @brief Performs 2D max pooling on the input tensor and stores the result in
 * the destination tensor.
 *
 * This function performs max pooling on the source tensor and stores the result
 * in the destination tensor. The pooling parameters (kernel size, strides,
 * padding) are specified in the `op_params` of the destination tensor.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The destination tensor where the result will be stored. The source
 * tensor is referenced by `dst->src[0]`.
 */
static void ggml_ponn_max_pool2d(ggml_backend_ponn_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t* opts = (const int32_t*)dst->op_params;
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];
    GGML_ASSERT(0);
}

void ggml_ponn_pool2d(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    const int32_t* opts = (const int32_t*)dst->op_params;
    enum ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    switch (op) {
        case GGML_OP_POOL_AVG:
            ggml_ponn_avg_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_MAX:
            ggml_ponn_max_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_COUNT:
            GGML_ABORT("fatal error");
            break;
    }
}
#if 0
void ponn_dup_no_contiguous_mmap(ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    ggml_tensor_extra_gpu *src_extra = (ggml_tensor_extra_gpu *)src0->extra;
    ggml_tensor_extra_gpu *dst_extra = (ggml_tensor_extra_gpu *)dst->extra;
    PONN_MEM_H input = src_extra->handle;
    PONN_MEM_H output = dst_extra->handle;

    char *mmap_src = nullptr;
    char *mmap_dst = nullptr;
    if(nnclMemMap(input, (void **)&mmap_src) != NNCL_STATUS_SUCCESS) {
        printf("nncl map failed, %d\n", __LINE__);
    }
    if(nnclMemMap(output, (void **)&mmap_dst) != NNCL_STATUS_SUCCESS) {
        printf("nncl map failed, %d\n", __LINE__);
    }

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        //copy by rows
        const size_t rs = ne00*nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const char * src0_ptr = ((char *) mmap_src + i01*nb01 + i02*nb02 + i03*nb03);
                    char * dst_ptr  = ((char *) mmap_dst + i01*nb1  + i02*nb2  + i03*nb3);
                    memcpy(dst_ptr, src0_ptr,rs);
                }
            }
        }
        return;
    }
    if (src0->type == dst->type && ggml_is_contiguous(dst)) {
        size_t id = 0;
        const size_t rs = ne00 * nb00;
        for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    const char * src0_ptr = (char *) mmap_src + i01*nb01 + i02*nb02 + i03*nb03;
                    memcpy((char *) mmap_dst + id, src0_ptr,rs);
                    id += rs;
                }
            }
        }
    }

    if(ggml_is_contiguous(dst) == false) {
        int64_t i10 = 0;
        int64_t i11 = 0;
        int64_t i12 = 0;
        int64_t i13 = 0;
        if (dst->type == GGML_TYPE_F16) {
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const char * src0_ptr = ((char *) mmap_src + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                            char * dst_ptr  = ((char *) mmap_dst + dst_extra->offset + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);
                            *(ggml_fp16_t *) dst_ptr = GGML_FP32_TO_FP16(*(const float *) src0_ptr);
                            if (++i10 == ne0) {
                                i10 = 0;
                                if (++i11 == ne1) {
                                    i11 = 0;
                                    if (++i12 == ne2) {
                                        i12 = 0;
                                        if (++i13 == ne3) {
                                            i13 = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            printf("%d not implements...\n",__LINE__);
            ggml_ponn_fallback(dst);
        }
    }
    nnclMemUnmap(input);
    nnclMemUnmap(output);
}
#endif
void ponn_dup_no_contiguous(ggml_tensor *dst) {
    ggml_tensor* src0 = dst->src[0];
    GGML_TENSOR_UNARY_OP_LOCALS

    ggml_tensor_extra_gpu *src_extra = (ggml_tensor_extra_gpu *)src0->extra;
    ggml_tensor_extra_gpu *dst_extra = (ggml_tensor_extra_gpu *)dst->extra;
    PONN_MEM_H input = src_extra->handle;
    PONN_MEM_H output = dst_extra->handle;

    size_t src_div = 1, dst_div = 1;

    // stride div
    ponn_utils_get_stride_div(src0->type, src_div);
    ponn_utils_get_stride_div(dst->type, dst_div);

    std::vector<int> inputDims = {ne03, ne02, ne01, ne00};
    std::vector<int> inputStrides = {nb03/src_div, nb02/src_div, nb01/src_div, nb00/src_div};
    std::vector<int> outputDims = {ne3, ne2, ne1, ne0};
    std::vector<int> outputStrides = {nb3/dst_div, nb2/dst_div, nb1/dst_div, nb0/dst_div};

    PONN_DATA_TYPE_E dtype = ponn_utils_get_data_type(dst->type);
    if(dtype == PONN_DATA_FLOAT && ponnGetInferenceDataType() == PONN_DATA_HALF)
        dtype = PONN_DATA_HALF;
    PONN_DATA_TYPE_E stype = ponn_utils_get_data_type(src0->type);
    if(stype == PONN_DATA_FLOAT && ponnGetInferenceDataType() == PONN_DATA_HALF)
        stype = PONN_DATA_HALF;
    ponnMemcpyNoContiguous(input, output, inputDims, inputStrides, outputDims, outputStrides,
                            stype, dtype,
                            src_extra->offset, dst_extra->offset);

    ponnFinish(src0, input);
    ponnFinish(dst, output);
}

void ggml_ponn_dup(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        PONN_MEM_H input = ponnPrepare(src0);
        PONN_MEM_H output = ponnPrepare(dst);
        if (src0->type == dst->type) {
            ponnMemcpy(output, 0, input, 0, ggml_nbytes(src0), DEVICE_TO_DEVICE);
        } else {
            PONN_DATA_TYPE_E dtype = ponn_utils_get_data_type(dst->type);
            if(dtype == PONN_DATA_FLOAT && ponnGetInferenceDataType() == PONN_DATA_HALF)
                dtype = PONN_DATA_HALF;
            PONN_DATA_TYPE_E stype = ponn_utils_get_data_type(src0->type);
            if(stype == PONN_DATA_FLOAT && ponnGetInferenceDataType() == PONN_DATA_HALF)
                stype = PONN_DATA_HALF;
            std::vector<int> dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
            ponnMemcpyEx(output, dtype, input, stype , ggml_nbytes(src0), dims, DEVICE_TO_DEVICE);

        }
    } else {
        // ggml_ponn_fallback(dst);

        ponn_dup_no_contiguous(dst);
        // ponn_dup_no_contiguous_mmap(dst);
    }
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.01f);
#endif
}

void ggml_ponn_rms_norm(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    if (0) {
        ggml_ponn_fallback(dst);
        return;
    }
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps > 0.0f);
    size_t one_tensor_n_bytes = src->ne[0] * ggml_element_size(src);
    PONN_MEM_H input0 = ponnPrepare(src);
    PONN_MEM_H input1 = ctx.get_one_tensor(one_tensor_n_bytes, ponnGetInferenceDataType());
    PONN_MEM_H output = ponnPrepare(dst);
    if (src1) {
        input1 = ponnPrepare(src1);
    }

    const int n = ggml_nrows(src);
    const int m = src->ne[0];
    std::vector<int> input0Dims = {n, m};
    std::vector<int> input1Dims  = {1, m};
    std::vector<int> outputDims  = {n, m};
    ponnRmsNorm(input0, input1, output, input0Dims, input1Dims, outputDims, eps);

    ponnFinish(src, input0);
    ponnFinish(dst, output);
    return;
}

void ggml_ponn_diag_mask(ggml_backend_ponn_context& ctx, ggml_tensor* dst,
                         float value) {
    ggml_tensor* src = dst->src[0];
    const int n_past = ((int32_t*)dst->op_params)[0];
    GGML_ASSERT(0);
}

void ggml_ponn_im2col(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];  // kernel
    ggml_tensor* src1 = dst->src[1];  // input

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;
    GGML_ASSERT(0);
}

void ggml_ponn_timestep_embedding(ggml_backend_ponn_context& ctx,
                                  ggml_tensor* dst) {
    const ggml_tensor* src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    const int dim = dst->op_params[0];
    const int max_period = dst->op_params[1];
    int half = dim / 2;
    GGML_ASSERT(0);
}

void ggml_ponn_cpy(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_ponn_dup(ctx, dst);
}

void ggml_ponn_soft_max(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];  // mask

    if (0) {
        ggml_ponn_fallback(dst);
        return;
    }
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H output = ponnPrepare(dst);

    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&max_bias, (float*)dst->op_params + 1, sizeof(float));

    std::vector<int> dims;
    ponn_utils_get_tensor_dims(dims, src0->ne);

    //mask
    PONN_MEM_H mask_tmp = nullptr;
    if (src1) {
        mask_tmp = ponnMallocBuf(ggml_nbytes(src0));
        PONN_MEM_H input1 = ponnPrepare(src1);
        if (max_bias <= 0) {
            std::vector<int> input1Dims = {1, 1, src0->ne[1], src0->ne[0]};
            ponnAdd(input0, input1, mask_tmp, dims, input1Dims, dims);

        } else {
            GGML_ASSERT(0);
        }
    }

    //softmax
    PONN_MEM_H soft_max_input = nullptr;
    mask_tmp ? soft_max_input = mask_tmp : soft_max_input = input0;

    ponnSoftmax(soft_max_input, output, dims);

    if (mask_tmp) {
        ponnFreeBuf(mask_tmp);
    }
    ponnFinish(src0, input0);
    ponnFinish(dst, output);
    return;
}

void ggml_ponn_get_rows(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS
    const int64_t nc = ne00;
    const int64_t nr = ggml_nelements(src1);
    GGML_ASSERT(ne0  == nc);
    GGML_ASSERT(ne02 == ne11);
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(ggml_nrows(dst) == nr);

    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    size_t type_div = 1;
    ponn_utils_get_stride_div(src0->type, type_div);

    std::vector<int> input0Dims = {ne03, ne02, ne01, ne00};
    std::vector<int> input1Dims = {ne13, ne12, ne11, ne10};
    std::vector<int> outputDims = {ne3, ne2, ne1, ne0};
    std::vector<int> input0Strides = {nb03/type_div, nb02/type_div, nb01/type_div, nb00/type_div};
    std::vector<int> input1Strides = {nb13, nb12, nb11, nb10};
    std::vector<int> outputStrides = {nb3/type_div, nb2/type_div, nb1/type_div, nb0/type_div};

    if (src0->type == GGML_TYPE_Q4_1) {
        input0Dims[3] /= QK4_1;
    }

    ponnGetRows(input0, input1, output, input0Dims, input0Strides, input1Dims,
                    input1Strides, outputDims, outputStrides, ponn_utils_get_data_type(src0->type));
    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
    ponnFinish(dst, output);

#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.0f);
#endif
    return;
}

/**
 * @brief Performs matrix multiplication with floating-point precision on
 * tensors using the PONN backend.
 *
 * This function performs matrix multiplication of the input tensor and the
 * weight tensor, handling broadcasting and transposing as needed, and stores
 * the result in the destination tensor `dst`.
 *
 * @param ctx The context for the PONN backend operations.
 * @param dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
static void ggml_ponn_mul_mat_fp(ggml_backend_ponn_context& ctx,
                                 ggml_tensor* dst) {

    ggml_tensor* weight = dst->src[0];  // weight
    ggml_tensor* input = dst->src[1];   // input
    ggml_tensor* bias = dst->src[2];    // bias

    // permute or type trans tmp buffer
    PONN_MEM_H w_for_type = nullptr; // weight tmp buffer for type trans: fp16->fp32
    PONN_MEM_H w_for_dims = nullptr; // weight tmp buffer for dims trans: no contingous -> contingous
    PONN_MEM_H i_for_dims = nullptr; // input tmp buffer for dims trans: no contingous -> contingous

    PONN_MEM_H input0 = ponnPrepare(input);
    PONN_MEM_H input1 = ponnPrepare(weight);
    PONN_MEM_H input2 = bias? ponnPrepare(bias): nullptr;
    PONN_MEM_H output = ponnPrepare(dst);

    // input permute
    if(ggml_is_permuted(input)) {
        size_t input_size = ggml_nbytes(input);
        i_for_dims = ponnMallocBuf(input_size);
        std::vector<int> dims = {input->ne[0], input->ne[1], input->ne[2], input->ne[3]};
        PONN_DATA_TYPE_E dtype = ponn_utils_get_data_type(input->type);
        if(dtype == PONN_DATA_FLOAT && ponnGetInferenceDataType() == PONN_DATA_HALF)
            dtype = PONN_DATA_HALF;
        ponnPermute(input0, i_for_dims, dtype, dims, {0, 2, 1, 3});
        input0 = i_for_dims;
    }

    // weight type
    size_t weight_size = ggml_nbytes(weight);
#if 0
    if(weight->type == GGML_TYPE_F16 && ponnGetInferenceDataType()== ZXNN_DATA_FLOAT) weight_size *= 2;
    if (weight->type == GGML_TYPE_F16) {
        w_for_type = ponnMallocBuf(weight_size);
        std::vector<int> dims;
        if(ggml_is_contiguous(weight) == true) {
            dims = {1, ggml_nelements(weight)};
        } else {
            dims = {1, weight_size/ggml_type_size(GGML_TYPE_F16)};
        }
        ponnMemcpyEx(w_for_type, ponn_utils_get_data_type(GGML_TYPE_F32), input1, ponn_utils_get_data_type(weight->type), weight_size/2, dims, DEVICE_TO_DEVICE);
        input1 = w_for_type;
    }
#endif
    // weight permute
    w_for_dims = ponnMallocBuf(weight_size);
    if(ggml_is_permuted(weight)) {
        std::vector<int> dims = {weight->ne[0], weight->ne[1], weight->ne[2], weight->ne[3]};
        PONN_DATA_TYPE_E dtype = ponn_utils_get_data_type(weight->type);
        if(dtype == PONN_DATA_FLOAT && ponnGetInferenceDataType() == PONN_DATA_HALF)
            dtype = PONN_DATA_HALF;
        ponnPermute(input1, w_for_dims, dtype, dims, {0, 2, 1, 3}); //for k-cache
        input1 = w_for_dims;
    }

    float alpha = 1.0f;
    PONN_MEM_H scale_tmp = nullptr;
    if(dst->op_params[1]) {
        memcpy(&alpha, (float *)dst->op_params + 1, sizeof(float));
        scale_tmp = ponnMallocBuf(ggml_nbytes(input));
        std::vector<int> dims;
        ponn_utils_get_tensor_dims(dims, input->ne);
        ponnScale(input0, scale_tmp, dims, alpha);
        alpha = 1.0f;
    }
    bool transB = true;
    int group = input->ne[2] / weight->ne[2];
    int n = input->ne[1] * group;
    int m = input->ne[0];
    int k = weight->ne[1];
    int batch = weight->ne[2];

    int input0Spatial = n * m;
    int input1Spatial = m * k;
    int outputSpatial = n * k;

    int input2Spatial = 0;
    if (bias) {
        input2Spatial = bias->ne[0]* bias->ne[1];
    }

    bool expanded = false;
    if(ponn_utils_is_padded_1d_0(weight)) {
        input1Spatial = weight->nb[3]/(batch*weight->nb[0]);
        expanded = true;
    }
    input0 = scale_tmp? scale_tmp:input0;
    if(bias) {
        GGML_ASSERT(expanded==false);
        ponnMulMatFpBias(input0, input1, input2, output,
                ponn_utils_get_data_type(input->type), ponn_utils_get_data_type(weight->type),
                ponn_utils_get_data_type(bias->type), ponn_utils_get_data_type(dst->type),
                input0Spatial, input1Spatial, input2Spatial, outputSpatial,
                batch, n, m, k, group, alpha);
    }else {
        ponnMulMatFp(input0, input1, output, ponn_utils_get_data_type(input->type), ponn_utils_get_data_type(weight->type),
                        ponn_utils_get_data_type(dst->type), input0Spatial, input1Spatial, outputSpatial,
                        batch, n, m, k, group, expanded, alpha);
    }


    if(w_for_type) ponnFreeBuf(w_for_type);
    if(w_for_dims) ponnFreeBuf(w_for_dims);
    if(i_for_dims) ponnFreeBuf(i_for_dims);
    if(scale_tmp)  ponnFreeBuf(scale_tmp);

    ponnFinish(input, input0);
    ponnFinish(weight, input1);
    ponnFinish(dst, output);
}

/**
 * @brief Performs matrix multiplication with quantized weights and
 * floating-point inputs using the PONN backend.
 *
 * This function performs matrix multiplication of the input tensor `src1` and
 * the weight tensor `src0`, handling broadcasting, transposing, and
 * quantization as needed, and stores the result in the destination tensor
 * `dst`.
 *
 * @param ctx The context for the PONN backend operations.
 * @param dst The destination tensor where the result of the matrix
 * multiplication will be stored.
 */
static void ggml_ponn_mul_mat_quant(ggml_backend_ponn_context& ctx,
                                   ggml_tensor* dst,
                                   const enum ggml_type type) {

    ggml_tensor* src0 = dst->src[0];  // weight
    ggml_tensor* src1 = dst->src[1];  // input
    ggml_tensor* src2 = dst->src[2];  // bias
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1));
    GGML_ASSERT(!ggml_is_permuted(src1));

    ggml_tensor_extra_gpu* w_extra =(ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu* dst_extra =(ggml_tensor_extra_gpu *) dst->extra;
    GGML_ASSERT(w_extra->extraPonnData.size() == 4);// weights/scales/mins/bias

    PONN_MEM_H input = ponnPrepare(src1);
    PONN_MEM_H weights = w_extra->extraPonnData[0];
    PONN_MEM_H scales = w_extra->extraPonnData[1];
    PONN_MEM_H mins = w_extra->extraPonnData[2];
    PONN_MEM_H bias = w_extra->extraPonnData[3];
    PONN_MEM_H output = ponnPrepare(dst);

    if (src2) {
        bias = ponnPrepare(src2);
    }

    int n = src1->ne[1], m = src1->ne[0], k = dst->ne[0];
    int blck_size = ggml_blck_size(type);
    PONN_DATA_TYPE_E quant_type = ponn_utils_get_data_type(src0->type);

    ponnMulMatQuant(input, weights, output, scales, mins, bias,
                        n, m ,k, blck_size, quant_type);

    ponnFinish(src1, input);
    ponnFinish(dst, output);
}

void ggml_ponn_mul_mat(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    if (0) {
        ggml_ponn_fallback(dst);
        return;
    }
    const enum ggml_type type = dst->src[0]->type;
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            ggml_ponn_mul_mat_fp(ctx, dst);
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            ggml_ponn_mul_mat_quant(ctx, dst, type);
            break;
        case GGML_TYPE_Q8_0:
            // ggml_ponn_mul_mat_quant(ctx, dst, type);
            ggml_ponn_fallback(dst);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}

// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static void ggml_rope_cache_init(
     float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

// void *ponn_sin = nullptr;
// void *ponn_cos = nullptr;
void update_ggml_ponn_rope_cache(float freq_base, int rotary_dim, int n_ctx_orig, float beta_fast, float beta_slow, float ext_factor, float attn_factor, float freq_scale, ggml_backend_ponn_context& ctx) {
    //copy from ggml.c: rope_f32 func
    const float theta_scale = powf(freq_base, -2.0f / rotary_dim);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(rotary_dim, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    float *rope_cache = (float *)malloc(sizeof(float) * n_ctx_orig * rotary_dim * 2);
    float *p = rope_cache;
    for(int i=0; i<n_ctx_orig; ++i){
        float pos = i;
        ggml_rope_cache_init(pos, freq_scale, nullptr, corr_dims, rotary_dim, ext_factor, attn_factor, p, 1.0f, theta_scale);
        p += rotary_dim * 2;
    }
    float *sin_cache = (float *)malloc(sizeof(float) * n_ctx_orig * rotary_dim);
    float *cos_cache = (float *)malloc(sizeof(float) * n_ctx_orig * rotary_dim);
    float *ps  = sin_cache;
    float *pc  = cos_cache;
    for(int i=0; i<n_ctx_orig * rotary_dim * 2; ++i) {
        if(i % 2 == 0) *(pc++) = rope_cache[i];
        else *(ps++) = rope_cache[i];
    }
    ctx.ponn_sin = ponnMallocBuf(sizeof(float) * (n_ctx_orig) * rotary_dim);
    ctx.ponn_cos = ponnMallocBuf(sizeof(float) * (n_ctx_orig) * rotary_dim);

    std::vector<int> sin_cos_dims ={1, n_ctx_orig * rotary_dim};
    ponnMemcpyEx(ctx.ponn_sin, ponnGetInferenceDataType(), sin_cache, PONN_DATA_FLOAT , \
                    n_ctx_orig * rotary_dim * sizeof(float), sin_cos_dims, HOST_TO_DEVICE);
    ponnMemcpyEx(ctx.ponn_cos, ponnGetInferenceDataType(), cos_cache, PONN_DATA_FLOAT , \
                    n_ctx_orig * rotary_dim * sizeof(float), sin_cos_dims, HOST_TO_DEVICE);
    free(rope_cache);
    free(sin_cache);
    free(cos_cache);
}


void ggml_ponn_rope(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    // Only test with LLAMA model.
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS

    const int rotary_dim = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    if(ctx.ponn_sin == nullptr && ctx.ponn_cos == nullptr) {
        //calculate sin data and cos data
        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
        memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
        memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
        memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
        memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
        memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
        memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
        update_ggml_ponn_rope_cache(freq_base, rotary_dim, n_ctx_orig, beta_fast, beta_slow, ext_factor, attn_factor, freq_scale, ctx);
    }
    GGML_ASSERT(ctx.ponn_sin);
    GGML_ASSERT(ctx.ponn_cos);

    std::vector<int> inputDims = {ne03, ne02, ne01, ne00};
    std::vector<int> outputDims = {ne3, ne2, ne1, ne0};
    std::vector<int> posIdDims = {1, 1, 1, ne10};
    std::vector<int> sinDims = {1, 1, rotary_dim, rotary_dim};
    std::vector<int> cosDims = {1, 1, rotary_dim, rotary_dim};

    PONN_MEM_H input = ponnPrepare(src0);
    PONN_MEM_H posId = ponnPrepare(src1);
    PONN_MEM_H sinData = ctx.ponn_sin;
    PONN_MEM_H cosData = ctx.ponn_cos;
    PONN_MEM_H output = ponnPrepare(dst);
    ponnRope(input, posId, sinData, cosData, output, inputDims, posIdDims, sinDims, cosDims, outputDims, rotary_dim, is_neox);
    ponnFinish(src0, input);
    ponnFinish(src1, posId);
    ponnFinish(dst, output);
#ifdef GGML_PONN_CHECK
    ggml_ponn_check(dst, 0.0001);
#endif
    return;
}

void ggml_ponn_mul(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    PONN_MEM_H input0 = ponnPrepare(src0);
    PONN_MEM_H input1 = ponnPrepare(src1);
    PONN_MEM_H output = ponnPrepare(dst);
    std::vector<int> input0Dims = {src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]};
    std::vector<int> input1Dims = {src1->ne[3], src1->ne[2], src1->ne[1], src1->ne[0]};
    std::vector<int> outputDims = {dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]};
    ponnMul(input0, input1, output, input0Dims, input1Dims, outputDims);

    ponnFinish(src0, input0);
    ponnFinish(src1, input1);
    ponnFinish(dst, output);
    return;
}

void ggml_ponn_div(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));
    GGML_ASSERT(0);
}

void ggml_ponn_unary(ggml_backend_ponn_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    //fallback
    if (0) {
        ggml_ponn_fallback(dst);
        return;
    }

    PONN_MEM_H input = ponnPrepare(src);
    PONN_MEM_H output = ponnPrepare(dst);

    const int n = ggml_nrows(src);
    const int m = src->ne[0];
    std::vector<int> dims = {n, m};
    switch (ggml_get_unary_op(dst)) {
        case GGML_UNARY_OP_GELU:
            ponnGelu(input, output, dims);
            break;
        case GGML_UNARY_OP_SILU:
            ponnSilu(input, output, dims);
            break;
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_HARDSIGMOID:
        case GGML_UNARY_OP_HARDSWISH:
            GGML_ASSERT(0);
            break;
        default:
            GGML_ASSERT(0);
    }

    ponnFinish(src, input);
    ponnFinish(dst, output);
    return;
}

void ggml_ponn_fallback(ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_TRANSPOSE) {
        return;
    }

    //printf("ggml_ponn_fallback %s \n", ggml_op_name(tensor->op));
    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];

    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ggml_ctx = ggml_init(iparams);

    struct ggml_tensor * src0_clone = nullptr;
    struct ggml_tensor * src1_clone = nullptr;
    struct ggml_tensor * src2_clone = nullptr;
    struct ggml_tensor * tensor_clone = nullptr;

    size_t src0_size;
    size_t src1_size;
    size_t src2_size;

    void * src0_buffer = nullptr;
    void * src1_buffer = nullptr;
    void * src2_buffer = nullptr;

    const uint32_t mode = GGML_SCALE_MODE_NEAREST;

    if (src0 != nullptr) {
        src0_clone = ggml_dup_tensor(ggml_ctx, src0);

        src0_size = ggml_nbytes(src0);

        src0_buffer = malloc(src0_size);
        src0_clone->data = src0_buffer;
        if (ggml_backend_buffer_is_host(src0->buffer)) {
            memcpy(src0_clone->data, src0->data, src0_size);
            memcpy(src0_clone->nb, src0->nb, sizeof(size_t) * GGML_MAX_DIMS);
        } else  {
            ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)src0->extra;
#if 0
            if (offset + src0_size >= buffer_gpu->size) {
                src0_size = buffer_gpu->size - offset;
            }
            ggml_vk_buffer_read(buffer_gpu, offset, src0_clone->data, src0_size);
#endif
            if(src0->type == GGML_TYPE_F32) {
                std::vector<int> src0_dims;
                ponn_utils_get_tensor_dims(src0_dims, src0->ne);
                ponnMemcpyEx(src0_clone->data, ponn_utils_get_data_type(src0->type),  extra->handle,ponnGetInferenceDataType() , src0_size, src0_dims, DEVICE_TO_HOST);
            }
            else {
                ponnMemcpy(src0_clone->data, 0, extra->handle, extra->offset, src0_size, DEVICE_TO_HOST);
            }
            memcpy(src0_clone->nb, src0->nb, sizeof(size_t) * GGML_MAX_DIMS);
        }
    }
    if (src1 != nullptr) {
        src1_clone = ggml_dup_tensor(ggml_ctx, src1);

        src1_size = ggml_nbytes(src1);

        src1_buffer = malloc(src1_size);
        src1_clone->data = src1_buffer;
        if (ggml_backend_buffer_is_host(src1->buffer)) {
            memcpy(src1_clone->data, src1->data, src1_size);
            memcpy(src1_clone->nb, src1->nb, sizeof(size_t) * GGML_MAX_DIMS);
        } else {
            ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)src1->extra;
#if 0
            if (offset + src0_size >= buffer_gpu->size) {
                src0_size = buffer_gpu->size - offset;
            }
            ggml_vk_buffer_read(buffer_gpu, offset, src0_clone->data, src0_size);
#endif
            if(src1->type == GGML_TYPE_F32) {
                std::vector<int> src1_dims;
                ponn_utils_get_tensor_dims(src1_dims, src1->ne);
                ponnMemcpyEx(src1_clone->data, ponn_utils_get_data_type(src1->type),  extra->handle,ponnGetInferenceDataType(), src1_size, src1_dims, DEVICE_TO_HOST);
            }
            else{
                ponnMemcpy(src1_clone->data, 0, extra->handle, extra->offset ,  src1_size, DEVICE_TO_HOST);
            }
            memcpy(src1_clone->nb, src1->nb, sizeof(size_t) * GGML_MAX_DIMS);
        }
    }

    if (src2 != nullptr) {
        src2_clone = ggml_dup_tensor(ggml_ctx, src2);

        src2_size = ggml_nbytes(src2);

        src2_buffer = malloc(src2_size);
        src2_clone->data = src2_buffer;
        if (ggml_backend_buffer_is_host(src2->buffer)) {
            memcpy(src2_clone->data, src2->data, src2_size);
            memcpy(src2_clone->nb, src2->nb, sizeof(size_t) * GGML_MAX_DIMS);
        } else  {
            ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)src2->extra;
#if 0
            if (offset + src0_size >= buffer_gpu->size) {
                src0_size = buffer_gpu->size - offset;
            }
            ggml_vk_buffer_read(buffer_gpu, offset, src0_clone->data, src0_size);
#endif
            if(src2->type == GGML_TYPE_F32) {
                std::vector<int> src2_dims;
                ponn_utils_get_tensor_dims(src2_dims, src2->ne);
                ponnMemcpyEx(src2_clone->data, ponn_utils_get_data_type(src2->type),  extra->handle,ponnGetInferenceDataType(), src2_size, src2_dims, DEVICE_TO_HOST);
            }
            else {
                ponnMemcpy(src2_clone->data, 0, extra->handle, extra->offset, src2_size, DEVICE_TO_HOST);
            }
            memcpy(src2_clone->nb, src2->nb, sizeof(size_t) * GGML_MAX_DIMS);
        }
    }

    if (tensor->op == GGML_OP_MUL_MAT) {
        tensor_clone = ggml_mul_mat(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_MUL_MAT_ID) {
        tensor_clone = ggml_mul_mat_id(ggml_ctx, src0_clone, src1_clone, src2_clone);
    } else if (tensor->op == GGML_OP_MUL) {
        tensor_clone = ggml_mul(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_DIV) {
        tensor_clone = ggml_div(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_CONCAT) {
        tensor_clone = ggml_concat(ggml_ctx, src0_clone, src1_clone, *(int *)tensor->op_params);
    } else if (tensor->op == GGML_OP_UPSCALE) {
        tensor_clone = ggml_interpolate(ggml_ctx, src0_clone, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], mode);
    } else if (tensor->op == GGML_OP_SCALE) {
        tensor_clone = ggml_scale(ggml_ctx, src0_clone, ((float *)tensor->op_params)[0]);
    } else if (tensor->op == GGML_OP_SQR) {
        tensor_clone = ggml_sqr(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_SIN) {
        tensor_clone = ggml_sin(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_COS) {
        tensor_clone = ggml_cos(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_CLAMP) {
        tensor_clone = ggml_clamp(ggml_ctx, src0_clone, ((float *)tensor->op_params)[0], ((float *)tensor->op_params)[1]);
    } else if (tensor->op == GGML_OP_PAD) {
        tensor_clone = ggml_pad(ggml_ctx, src0_clone, tensor->ne[0] - src0_clone->ne[0], tensor->ne[1] - src0_clone->ne[1], tensor->ne[2] - src0_clone->ne[2], tensor->ne[3] - src0_clone->ne[3]);
    } else if (tensor->op == GGML_OP_REPEAT) {
        tensor_clone = ggml_repeat(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_ADD) {
        tensor_clone = ggml_add(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_ACC) {
        tensor_clone = ggml_acc(ggml_ctx, src0_clone, src1_clone, tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
    } else if (tensor->op == GGML_OP_NORM) {
        tensor_clone = ggml_norm(ggml_ctx, src0_clone, *(float *)tensor->op_params);
    } else if (tensor->op == GGML_OP_GROUP_NORM) {
        tensor_clone = ggml_group_norm(ggml_ctx, src0_clone, *(int *)tensor->op_params, ((float *)tensor->op_params)[1]);
    } else if (tensor->op == GGML_OP_RMS_NORM) {
        tensor_clone = ggml_rms_norm(ggml_ctx, src0_clone, *(float *)tensor->op_params);
    } else if (tensor->op == GGML_OP_SOFT_MAX) {
        if (src1 != nullptr) {
            tensor_clone = ggml_soft_max_ext(ggml_ctx, src0_clone, src1_clone, ((float *)tensor->op_params)[0], ((float *)tensor->op_params)[1]);
        } else {
            tensor_clone = ggml_soft_max(ggml_ctx, src0_clone);
        }
    } else if (tensor->op == GGML_OP_DIAG_MASK_INF) {
        tensor_clone = ggml_diag_mask_inf(ggml_ctx, src0_clone, *(int *)tensor->op_params);
    } else if (tensor->op == GGML_OP_ROPE) {
        const int n_dims      = ((int32_t *) tensor->op_params)[1];
        const int mode        = ((int32_t *) tensor->op_params)[2];
        //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
        const int n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
        const float freq_base       = ((float *) tensor->op_params)[5];
        const float freq_scale      = ((float *) tensor->op_params)[6];
        const float ext_factor      = ((float *) tensor->op_params)[7];
        const float attn_factor     = ((float *) tensor->op_params)[8];
        const float beta_fast       = ((float *) tensor->op_params)[9];
        const float beta_slow       = ((float *) tensor->op_params)[10];
        tensor_clone = ggml_rope_ext(ggml_ctx, src0_clone, src1_clone, src2_clone, n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    } else if (tensor->op == GGML_OP_UNARY) {
        switch (ggml_get_unary_op(tensor)) {
        case GGML_UNARY_OP_SILU:
            tensor_clone = ggml_silu(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_GELU:
            tensor_clone = ggml_gelu(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_GELU_QUICK:
            tensor_clone = ggml_gelu_quick(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_RELU:
            tensor_clone = ggml_relu(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_TANH:
            tensor_clone = ggml_tanh(ggml_ctx, src0_clone);
            break;
        default:
            std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
            GGML_ABORT("fatal error");
        }
    } else if (tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_DUP) {
        if (src1 == nullptr) {
            tensor_clone = ggml_dup(ggml_ctx, src0_clone);
            tensor_clone->type = tensor->type;
        } else {
            tensor_clone = ggml_cpy(ggml_ctx, src0_clone, src1_clone);
        }
    } else if (tensor->op == GGML_OP_CONT) {
        tensor_clone = ggml_cont_4d(ggml_ctx, src0_clone, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    } else if (tensor->op == GGML_OP_RESHAPE) {
        tensor_clone = ggml_reshape_4d(ggml_ctx, src0_clone, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    } else if (tensor->op == GGML_OP_VIEW) {
        tensor_clone = ggml_view_4d(ggml_ctx, src0_clone, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
    } else if (tensor->op == GGML_OP_PERMUTE) {
        int32_t * params = (int32_t *)tensor->op_params;
        tensor_clone = ggml_permute(ggml_ctx, src0_clone, params[0], params[1], params[2], params[3]);
    } else if (tensor->op == GGML_OP_TRANSPOSE) {
        tensor_clone = ggml_transpose(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_GET_ROWS) {
        tensor_clone = ggml_get_rows(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_ARGSORT) {
        tensor_clone = ggml_argsort(ggml_ctx, src0_clone, (ggml_sort_order) *(int *)tensor->op_params);
    } else if (tensor->op == GGML_OP_SUM_ROWS) {
        tensor_clone = ggml_sum_rows(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_IM2COL) {
        const int32_t s0 = tensor->op_params[0];
        const int32_t s1 = tensor->op_params[1];
        const int32_t p0 = tensor->op_params[2];
        const int32_t p1 = tensor->op_params[3];
        const int32_t d0 = tensor->op_params[4];
        const int32_t d1 = tensor->op_params[5];

        const bool is_2D = tensor->op_params[6] == 1;
        tensor_clone = ggml_im2col(ggml_ctx, src0_clone, src1_clone, s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
    } else if (tensor->op == GGML_OP_TIMESTEP_EMBEDDING) {
        const int32_t dim = tensor->op_params[0];
        const int32_t max_period = tensor->op_params[1];
        tensor_clone = ggml_timestep_embedding(ggml_ctx, src0_clone, dim, max_period);
    } else if (tensor->op == GGML_OP_LEAKY_RELU) {
        const float * op_params = (const float *)tensor->op_params;
        tensor_clone = ggml_leaky_relu(ggml_ctx, src0_clone, op_params[0], false);
    } else {
        std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
        GGML_ABORT("fatal error");
    }

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_clone);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 8);

    size_t tensor_size = ggml_nbytes(tensor_clone);
    ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)tensor->extra;
    if(tensor_clone->type == GGML_TYPE_F32) {
        std::vector<int> tensor_clone_dims;
        ponn_utils_get_tensor_dims(tensor_clone_dims, tensor_clone->ne);
        ponnMemcpyEx( extra->handle,  ponnGetInferenceDataType(),  tensor_clone->data,ponn_utils_get_data_type(tensor_clone->type),tensor_size, tensor_clone_dims, HOST_TO_DEVICE);
    } else {
        ponnMemcpy(extra->handle, extra->offset, tensor_clone->data, 0,  tensor_size, HOST_TO_DEVICE);
    }

    if (src0 != nullptr) {
        free(src0_buffer);
    }
    if (src1 != nullptr) {
        free(src1_buffer);
    }
    if (src2 != nullptr) {
        free(src2_buffer);
    }

    ggml_free(ggml_ctx);
}
