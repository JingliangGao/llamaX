#include "ggml-ponn.h"

#include <stdarg.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "ggml-backend-impl.h"
#include "ggml-ponn/ops.h"
#include "ggml-ponn/common.h"
#include "ggml-ponn/ponn.h"
#include "utils.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#define GGML_PONN_NAME "PONN"

#define GGML_PONN_TRACE() //printf("PONN: %s \n", __FUNCTION__)
#define GGML_PONN_NOT_SUPPORT() GGML_LOG_ERROR("PONN: %s not support yest!!!!!\n", __FUNCTION__)

/**
 * @brief Handles PONN errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] void ggml_ponn_error(const char* stmt, const char* func,
                                  const char* file, int line, const char* msg) {
    int32_t id = -1;
    ponnGetDevice(&id);
    GGML_LOG_ERROR("PONN error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func,
            file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("PONN error");
}

/**
 * @brief Sets the device to be used by PONN.
 *
 * @param device The device ID to set.
 */
void ggml_ponn_set_device(const int32_t device) {
    GGML_PONN_TRACE();
    ponnSetDevice(device);
}

/**
 * @brief Retrieves the current device ID.
 *
 * @return The current device ID.
 */
int32_t ggml_ponn_get_device() {
    GGML_PONN_TRACE();
    int32_t id;
    ponnGetDevice(&id);
    return id;
}

/**
 * @brief Initialize the PONN device information.
 *
 * This function initializes the PONN device information by obtaining the
 * device count and setting the memory allocation granularity for each device.
 *
 * @return A structure containing the device information.
 */
static ggml_ponn_device_info ggml_ponn_init() {
    GGML_PONN_TRACE();
    ggml_ponn_device_info info = {};

    ponnInit();
    ponnGetDeviceCount(&info.device_count);
    GGML_LOG_INFO("ggml_ponn_init %d \n", info.device_count);
    GGML_ASSERT(info.device_count <= GGML_PONN_MAX_DEVICES);

    for (int id = 0; id < info.device_count; ++id) {
#if 0
        aclrtPhysicalMemProp prop = {};
        prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        prop.memAttr = ACL_HBM_MEM_HUGE;
        prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = id;
        prop.reserve = 0;
        PONN_CHECK(aclrtMemGetAllocationGranularity(
            &prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            &info.devices[id].vmm_granularity));
#endif
    }
    // TODO: add more device info later.
    return info;
}

/**
 * @brief Retrieve the PONN device information.
 *
 * This function returns a reference to a structure containing the PONN device
 * information. The device information is initialized once and reused on
 * subsequent calls.
 *
 * @return A reference to the structure containing the device information.
 */
const ggml_ponn_device_info& ggml_ponn_info() {
    GGML_PONN_TRACE();
    static ggml_ponn_device_info info = ggml_ponn_init();
    return info;
}

static void * const ponn_ptr_base = (void *)(uintptr_t) 0x1000;

// ponn buffer
/**
 * @brief Context for managing a PONN buffer associated with a specific device.
 *
 * This structure holds information about a PONN buffer, including the device
 * ID, device pointer, and a name derived from GGML_PONN_NAME and the device ID.
 */
struct ggml_backend_ponn_buffer_context {
    int32_t device;  ///< The device ID associated with this buffer context.
    void* dev_ptr = nullptr;  ///< Pointer to the device memory allocated for the buffer.
    void* handle = nullptr;  //< Handle of the device memory allocated for the buffer.
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    /**
     * @brief Constructor to initialize the PONN buffer context.
     *
     * @param device The device ID associated with this buffer context.
     * @param dev_ptr Pointer to the device memory allocated for the buffer.
     */
    ggml_backend_ponn_buffer_context(int32_t device, void * handle, void* dev_ptr)
        : device(device),
          dev_ptr(dev_ptr),
          handle(handle){}

    /**
     * @brief Destructor to free the device memory allocated for the buffer.
     */
    ~ggml_backend_ponn_buffer_context() {
        if (handle) {
            ponnFree(handle);
        }
    }
};

/**
 * @brief Retrieve the name associated with a PONN buffer.
 *
 * This function returns the name of a PONN buffer, which is stored in the
 * context of the buffer.
 *
 * @param buffer The PONN buffer whose name is to be retrieved.
 * @return A pointer to a C-string containing the name of the buffer.
 */

static const char* ggml_backend_ponn_buffer_get_name(
    ggml_backend_buffer_t buffer) {
    GGML_PONN_TRACE();
    return "PONN";

    GGML_UNUSED(buffer);
}

/**
 * @brief Check if a buffer is a PONN buffer.
 *
 * This function checks if a given buffer is a PONN buffer by comparing its
 * `get_name` function pointer to `ggml_backend_ponn_buffer_get_name`.
 *
 * @param buffer The buffer to check.
 * @return true if the buffer is a PONN buffer, false otherwise.
 */
static bool ggml_backend_buft_is_ponn(ggml_backend_buffer_type_t buft);
static bool ggml_backend_buffer_is_ponn(
    ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_ponn(buffer->buft);
}

/**
 * @brief Free resources associated with a PONN buffer.
 *
 * This function frees the resources associated with a PONN buffer, including
 * its context.
 *
 * @param buffer The PONN buffer to free.
 */
static void ggml_backend_ponn_buffer_free_buffer(
    ggml_backend_buffer_t buffer) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_buffer_context* ctx =
        (ggml_backend_ponn_buffer_context*)buffer->context;
    delete ctx;
}

/**
 * @brief Retrieve the base pointer of a PONN buffer.
 *
 * This function returns the base pointer of a PONN buffer, which points to the
 * device memory allocated for the buffer.
 *
 * @param buffer The PONN buffer whose base pointer is to be retrieved.
 * @return A pointer to the base of the device memory allocated for the buffer.
 */
static void* ggml_backend_ponn_buffer_get_base(
    ggml_backend_buffer_t buffer) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_buffer_context* ctx =
        (ggml_backend_ponn_buffer_context*)buffer->context;
    return ctx->dev_ptr;
}

/**
 * @brief Transform quantized Q4.1 tensor data into a format suitable for PONN
 * processing.
 *
 * This function transforms quantized Q4.1 tensor data into a format suitable
 * for PONN processing. It extracts quantization values and scales from the
 * source data and prepares them in a format expected by PONN operations.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data(host) in Q4.1 format.
 * @param w_dst Pointer to the int4 weights buffer where transformed data will be stored.
 * @param s_dst Pointer to the scales buffer where transformed data will be stored.
 * @param m_dst Pointer to the mins buffer where transformed data will be stored.
 * @param w_rows rows of weight matrix
 * @param w_clos clos of weight matrix
 */
 static void ggml_backend_ponn_transform_q4_1(ggml_tensor* tensor,
                                            const void* src,
                                            uint8_t* w_dst,
                                            uint16_t* s_dst,
                                            uint16_t* m_dst,
                                            int w_rows,
                                            int w_clos) {
    GGML_PONN_TRACE();
    int64_t n_elems = ggml_nelements(tensor);
    int64_t groups = n_elems / QK4_1;
    size_t quant_bytes = n_elems * sizeof(uint8_t) / 2;

    uint8_t* quant_offset = w_dst;

    for (int i = 0; i < groups; i++) {
        const block_q4_1* group =
            (const block_q4_1*)((const char*)src + i * sizeof(block_q4_1));
        *s_dst = group->d;
        s_dst++;

        *m_dst = group->m;
        m_dst++;

        for (int j = 0; j < QK4_1 / 2; j += 2) {
            (*quant_offset) = (group->qs[j]  << 4);
            (*quant_offset) |= (group->qs[j + 1]& 0x0F);
            quant_offset++;
        }

        // 16-31
        for (int j = 0; j < QK4_1 / 2; j += 2) {
            (*quant_offset) = (group->qs[j] & 0xF0);
            (*quant_offset) |= (group->qs[j + 1] >> 4);
            quant_offset++;
        }
    }
}

/**
 * @brief Transform tensor data based on its type for PONN processing.
 *
 * This function transforms tensor data based on its quantization type for PONN
 * processing. It dispatches the transformation based on the tensor's type to
 * specialized functions handling Q4.1 formats.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data to be transformed.
 */
static void ggml_backend_ponn_transform(ggml_tensor* tensor,
                                                  const void* src) {
    GGML_PONN_TRACE();
    ggml_tensor_extra_gpu *extrad = (ggml_tensor_extra_gpu *)tensor->extra;
    int64_t n_elems = ggml_nelements(tensor);
    const enum ggml_type type = tensor->type;
    size_t blck_size = ggml_blck_size(type);
    int64_t groups = n_elems / blck_size;
    int w_rows = tensor->ne[1];
    int w_clos = tensor->ne[0];

    uint16_t* scales = new uint16_t[groups];
    uint16_t* mins = new uint16_t[groups];
    float* bias = new float[w_rows]();
    uint8_t* weights =  new uint8_t [n_elems * sizeof(uint8_t) / 2];

    switch (tensor->type) {
        case GGML_TYPE_Q4_1:
            ggml_backend_ponn_transform_q4_1(tensor, src, weights, scales, mins, w_rows, w_clos);
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        default:
            GGML_ASSERT(0);
            break;
    }

    //mins and scales tranpose
    int s_rows = w_rows;
    int s_clos = w_clos / QK4_1;
    std::vector<uint16_t> s_tmp(scales, scales + groups);
    std::vector<uint16_t> m_tmp(mins, mins + groups);
    for (int i = 0; i < s_rows; i++) {
        for (int j = 0; j < s_clos; j++) {
            scales[j * s_rows + i] = s_tmp[i * s_clos + j];
            mins[j * s_rows + i] = m_tmp[i * s_clos + j];
        }
    }

    //weight tranpose and pack
    std::vector<uint8_t> dataT(weights, weights + n_elems * sizeof(uint8_t) / 2);
#define UCHAR_16 16
#define WEIGHT_DATA  UCHAR_16
    for (int l_ = 0, l, i_, i, j = 0; j < w_rows; ++j) {
        for (i_ = 0, i = 0; i < w_clos / 2; i += WEIGHT_DATA, ++i_) {
            for (l = 0; l < WEIGHT_DATA; ++l)
                weights[i_ * (w_rows * WEIGHT_DATA) + l_ + l] = dataT[w_clos / 2 * j + i + l];
        }
        l_ += WEIGHT_DATA;
    }

    //
    ggml_backend_ponn_buffer_context *ctx =
        (ggml_backend_ponn_buffer_context *)tensor->buffer->context;
    void *dev_ptr = tensor->buffer->iface.get_base(tensor->buffer);
    size_t buffer_offset = ((char *)tensor->data - (char *)dev_ptr);
    size_t offset = buffer_offset;

    void* weightsDev;
    size_t weightsSize = n_elems * sizeof(uint8_t) / 2;
    offset = ((offset - 1) / PONN_BUFFER_ALIGNMENT + 1) * PONN_BUFFER_ALIGNMENT;
    if (offset - buffer_offset + weightsSize <= ggml_nbytes(tensor)) {
        weightsDev = ponnMallocSubBuf(ctx->handle, offset, weightsSize);
    } else {
        weightsDev = ponnMallocBuf(weightsSize);
    }
    ponnMemcpy(weightsDev, 0, weights, 0, weightsSize, HOST_TO_DEVICE);
    extrad->extraPonnData.push_back((void*)weightsDev);
    offset += weightsSize;

    void* scalesDev;
    size_t scalesSize = groups * sizeof(uint16_t);
    offset = ((offset - 1) / PONN_BUFFER_ALIGNMENT + 1) * PONN_BUFFER_ALIGNMENT;
    if (offset - buffer_offset + scalesSize <= ggml_nbytes(tensor)) {
        scalesDev = ponnMallocSubBuf(ctx->handle, offset, scalesSize);
    } else {
        scalesDev = ponnMallocBuf(scalesSize);
    }
    ponnMemcpy(scalesDev, 0, scales, 0, scalesSize, HOST_TO_DEVICE);
    extrad->extraPonnData.push_back((void*)scalesDev);
    offset += scalesSize;

    void* minsDev;
    size_t minsSize = groups * sizeof(uint16_t);
    offset = ((offset - 1) / PONN_BUFFER_ALIGNMENT + 1) * PONN_BUFFER_ALIGNMENT;
    if (offset - buffer_offset + minsSize <= ggml_nbytes(tensor)) {
        minsDev = ponnMallocSubBuf(ctx->handle, offset, minsSize);
    } else {
        minsDev = ponnMallocBuf(minsSize);
    }
    ponnMemcpy(minsDev, 0, mins, 0, minsSize, HOST_TO_DEVICE);
    extrad->extraPonnData.push_back((void*)minsDev);

    void* biasDev = ponnMallocBuf(w_rows * sizeof(float));
    float value = 0;
    memset(bias, 0, w_rows * sizeof(float));
    std::vector<int> bias_dims = {1, w_rows};
    ponnMemcpyEx(biasDev, ponnGetInferenceDataType(), bias, PONN_DATA_FLOAT , w_rows * sizeof(float), bias_dims, HOST_TO_DEVICE);
    extrad->extraPonnData.push_back(biasDev);

    if(weights) delete[] weights;
    if(scales) delete[] scales;
    if(mins) delete[] mins;
    if(bias) delete[] bias;
}

/**
 * @brief Transform PONN processed data back into tensor data based on its type.
 *
 * This function transforms PONN processed data back into tensor data based on
 * its quantization type for Q4.0 and Q8.0 formats. It dispatches the
 * transformation based on the tensor's type to specialized functions.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data containing PONN processed data.
 * @param dst Pointer to the destination buffer where transformed tensor data
 * will be stored.
 */
static void ggml_backend_ponn_transform_back(
    const ggml_tensor* tensor, void* src, void* dst) {
    GGML_PONN_TRACE();
    switch (tensor->type) {
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            GGML_ASSERT(0);
            break;
        default:
            GGML_ASSERT(0);
            break;
    }
}

/**
 * @brief Check if transformation is needed for a given tensor type.
 *
 * This function checks if transformation is needed for a given tensor type
 * to prepare data for PONN processing.
 *
 * @param type The tensor type to check.
 * @return true if transformation is needed, false otherwise.
 */
static bool need_transform(ggml_type type) {
    GGML_PONN_TRACE();
    switch (type) {
        case GGML_TYPE_Q4_1:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Initialize a tensor using data from a PONN buffer.
 *
 * This function initializes a tensor using data from a PONN buffer.
 * It handles special cases such as views and quantization.
 *
 * @param buffer The PONN buffer from which to initialize the tensor.
 * @param tensor Pointer to the tensor to be initialized.
 */
static enum ggml_status ggml_backend_ponn_buffer_init_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor* tensor) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_buffer_context *ctx =
        (ggml_backend_ponn_buffer_context *)buffer->context;

    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        //tensor->backend = tensor->view_src->backend;
        tensor->extra = tensor->view_src->extra;
        return GGML_STATUS_SUCCESS;
    }
    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
    void *dev_ptr = buffer->iface.get_base(buffer);
    size_t offset = ((char *)tensor->data - (char *)dev_ptr);
    extra->buf_handle = ctx->handle;
    extra->buf_offset = offset;
    size_t size = ggml_nbytes(tensor);
    //buf addres must be PONN_BUFFER_ALIGNMENT bytes alignment
    size_t offset_aligned = offset % PONN_BUFFER_ALIGNMENT;
    extra->handle = ponnMallocSubBuf(ctx->handle, offset - offset_aligned, size + offset_aligned);
    extra->offset = offset_aligned;
    ctx->tensor_extras.push_back(extra);
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}

// TODO: need handle tensor which has paddings.
/**
 * @brief Set tensor data in a PONN buffer.
 *
 * This function sets tensor data in a PONN buffer, handling transformations
 * if needed based on the tensor's type.
 *
 * @param buffer The PONN buffer where the tensor data will be set.
 * @param tensor Pointer to the tensor whose data will be set.
 * @param data Pointer to the source data to be copied into the tensor.
 * @param offset Offset in the tensor data from where to start copying.
 * @param size Size of the data to be copied, in bytes.
 */
static void ggml_backend_ponn_buffer_set_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor *tensor, const void *data,
    size_t offset, size_t size) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_buffer_context *ctx =
        (ggml_backend_ponn_buffer_context *)buffer->context;

    ggml_ponn_set_device(ctx->device);
    // For ponn, synchronous functions use this default stream.
    if(need_transform(tensor->type)) {
        ggml_backend_ponn_transform(tensor, data);
    } else {
        ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)tensor->extra;
        if(tensor->type == GGML_TYPE_F32){
            std::vector<int> tensor_dims;
            ponn_utils_get_tensor_dims(tensor_dims, tensor->ne);
            ponnMemcpyEx(extra->handle, ponnGetInferenceDataType(), data, ponn_utils_get_data_type(tensor->type), size, tensor_dims, HOST_TO_DEVICE);
        } else {
            ponnMemcpy(extra->handle, extra->offset + offset, data, 0, size, HOST_TO_DEVICE);
        }
    }
}

/**
 * @brief Get tensor data from a PONN buffer.
 *
 * This function retrieves tensor data from a PONN buffer, handling
 * transformations if needed based on the tensor's type.
 *
 * @param buffer The PONN buffer from which to retrieve tensor data.
 * @param tensor Pointer to the tensor whose data will be retrieved.
 * @param data Pointer to the destination buffer where the tensor data will be
 * copied.
 * @param offset Offset in the destination buffer where to start copying.
 * @param size Size of the data to be copied, in bytes.
 */
static void ggml_backend_ponn_buffer_get_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* data,
    size_t offset, size_t size) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_buffer_context* ctx =
        (ggml_backend_ponn_buffer_context*)buffer->context;
    ggml_ponn_set_device(ctx->device);
    if (!need_transform(tensor->type)) {
        ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)tensor->extra;
        if(tensor->type == GGML_TYPE_F32) {
            std::vector<int> tensor_dims;
            ponn_utils_get_tensor_dims(tensor_dims, tensor->ne);
            ponnMemcpyEx(data, ponn_utils_get_data_type(tensor->type),  extra->handle, ponnGetInferenceDataType(),  size, tensor_dims, DEVICE_TO_HOST);
        }else{
            ponnMemcpy(data, 0, extra->handle, extra->offset + offset,  size, DEVICE_TO_HOST);
        }
    } else {
        GGML_ASSERT(0);
    }
}

/**
 * @brief Copy tensor data between PONN buffers if possible.
 *
 * This function copies tensor data between PONN buffers if the source and
 * destination buffers are PONN buffers and they meet the necessary conditions
 * (same device or devices can access each other).
 *
 * @param buffer The destination PONN buffer where the tensor data will be
 * copied.
 * @param src Pointer to the source tensor whose data will be copied.
 * @param dst Pointer to the destination tensor where the data will be copied.
 * @return true if the copy operation succeeded, false otherwise.
 */
static bool ggml_backend_ponn_buffer_cpy_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* src, ggml_tensor* dst) {
    GGML_PONN_TRACE();
    GGML_PONN_NOT_SUPPORT();
#if 0
    if (ggml_backend_buffer_is_ponn(src->buffer)) {
        ggml_backend_ponn_buffer_context* src_ctx =
            (ggml_backend_ponn_buffer_context*)src->buffer->context;
        ggml_backend_ponn_buffer_context* dst_ctx =
            (ggml_backend_ponn_buffer_context*)buffer->context;
        size_t memcpy_size = ggml_nbytes(src);
        // Same device.
        if (src_ctx->device == dst_ctx->device) {
            PONN_CHECK(aclrtMemcpy((char*)dst->data, memcpy_size,
                                  (const char*)src->data, memcpy_size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE));
            return true;
        } else {
            // Different device but can access by peer.
            int32_t canAccessPeer = 0;
            PONN_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, src_ctx->device,
                                               dst_ctx->device));
            if (canAccessPeer) {
                ggml_ponn_set_device(src_ctx->device);
                PONN_CHECK(aclrtDeviceEnablePeerAccess(dst_ctx->device, 0));
                PONN_CHECK(aclrtMemcpy((char*)dst->data, memcpy_size,
                                      (const char*)src->data, memcpy_size,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE));
                return true;
            }
        }
    }
#endif
    return false;
}

/**
 * @brief Clear a PONN buffer by setting all its memory to a specified value.
 *
 * This function clears a PONN buffer by setting all its memory to a specified
 * value.
 *
 * @param buffer The PONN buffer to be cleared.
 * @param value The value to which each byte in the buffer will be set.
 */
static void ggml_backend_ponn_buffer_clear(
    ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_buffer_context* ctx =
        (ggml_backend_ponn_buffer_context*)buffer->context;
    ggml_ponn_set_device(ctx->device);
    ponnMemset(ctx->handle, 0, buffer->size, &value, sizeof(uint8_t));
}

/**
 * @brief Reset a PONN buffer by free the sub buffers.
 *
 * This function reset a PONN buffer by free all its sub buffer
 *
 * @param buffer The PONN buffer to be cleared.
 */
static void ggml_backend_ponn_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_ponn_buffer_context * ctx = (ggml_backend_ponn_buffer_context *) buffer->context;
    for (auto * extra : ctx->tensor_extras) {
        if (extra->handle != ctx->handle) {
            ponnFreeSubBuf(extra->handle);
            extra->handle = nullptr;
        }
        delete extra;
    }
    ctx->tensor_extras.clear();
}

/**
 * @brief Interface for a PONN buffer in the backend.
 *
 * This structure defines function pointers to operations that can be performed
 * on a PONN buffer within the backend.
 */
static ggml_backend_buffer_i ggml_backend_ponn_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_ponn_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_ponn_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_ponn_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_ponn_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_ponn_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_ponn_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_ponn_buffer_clear,
    /* .reset           = */ ggml_backend_ponn_buffer_reset,
};

// ponn buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
struct ggml_backend_ponn_buffer_type_context {
    int32_t device; /**< Device identifier associated with the buffer context. */
    std::string name; /**< Name associated with the buffer context. */
};

/**
 * @brief Retrieves the name associated with a PONN buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * PONN buffer type context.
 *
 * @param buft Pointer to the buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
static const char* ggml_backend_ponn_buffer_type_name(
    ggml_backend_buffer_type_t buft) {
    GGML_PONN_TRACE();
    return "PONN";

    GGML_UNUSED(buft);
}

/**
 * @brief Allocates a new PONN buffer of the specified type and size.
 *
 * This function allocates a new PONN buffer on the specified device with the
 * given size.
 *
 * @param buft Pointer to the buffer type context.
 * @param size Size in bytes of the buffer to allocate.
 * @return Pointer to the allocated buffer, or nullptr if allocation fails.
 */
static ggml_backend_buffer_t
ggml_backend_ponn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_buffer_type_context* buft_ctx =
        (ggml_backend_ponn_buffer_type_context*)buft->context;

    ggml_ponn_set_device(buft_ctx->device);
    size = std::max(size, (size_t)1);

    void* handle = nullptr;
    void* dev_ptr = nullptr;
    handle = ponnMallocBuf(size);
    dev_ptr = ponn_ptr_base;
    if (handle == nullptr) {
        GGML_LOG_ERROR(
            "%s: allocating %.2f MiB on device %d: ponnMalloc failed: %s\n",
            __func__, size / 1024.0 / 1024.0, buft_ctx->device,
            "malloc fail");
        return nullptr;
    }
    ggml_backend_ponn_buffer_context* ctx =
        new ggml_backend_ponn_buffer_context(buft_ctx->device, handle, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_ponn_buffer_interface,
                                    ctx, size);
}

/**
 * @brief Retrieves the memory alignment requirement for PONN buffers of this
 * type.
 *
 * This function returns the alignment requirement in bytes for memory allocated
 * by the PONN buffer type.
 *
 * @param buft Pointer to the buffer type context (unused in this
 * implementation).
 * @return The alignment requirement in bytes (fixed at 256 bytes for PONN
 * buffers).
 */
static size_t ggml_backend_ponn_buffer_type_get_alignment(
    ggml_backend_buffer_type_t buft) {
    GGML_PONN_TRACE();
    return PONN_BUFFER_ALIGNMENT;

    GGML_UNUSED(buft);
}

/**
 * @brief Calculates the allocation size required for a tensor in a PONN buffer.
 *
 * Computes the total allocation size needed for storing the tensor's data in a
 * PONN buffer, considering any necessary padding or adjustments for quantized
 * types.
 *
 * @param buft Pointer to the buffer type context (unused in this
 * implementation).
 * @param tensor Pointer to the tensor for which the allocation size is
 * calculated.
 * @return The total allocation size in bytes required for the tensor in the
 * PONN buffer.
 */
static size_t ggml_backend_ponn_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {
    GGML_PONN_TRACE();
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // TODO: not support quantized yet.
    // TODO: consider un-continue tensor.
#if 0
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(
                tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }
#endif
    return size;

    GGML_UNUSED(buft);
}

/**
 * @brief Interface for managing PONN buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for PONN buffer types in the GGML backend.
 */
static ggml_backend_buffer_type_i ggml_backend_ponn_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_ponn_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_ponn_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_ponn_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_ponn_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

/**
 * @brief Retrieves the PONN buffer type for a specified device.
 *
 * This function initializes and returns the buffer type interface associated
 * with the given device. It ensures thread-safe access using a mutex.
 *
 * @param device The device index for which to retrieve the buffer type.
 * @return A pointer to the buffer type interface for the specified device, or
 * nullptr if the device index is out of range.
 */
ggml_backend_buffer_type_t
ggml_backend_ponn_buffer_type(int32_t device) {
    GGML_PONN_TRACE();
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_ponn_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type
        ggml_backend_ponn_buffer_types[GGML_PONN_MAX_DEVICES];

    static bool ggml_backend_ponn_buffer_type_initialized = false;

    if (!ggml_backend_ponn_buffer_type_initialized) {
        for (int32_t i = 0; i < ggml_ponn_info().device_count; i++) {
            ggml_backend_ponn_buffer_types[i] = {
                /* .iface    = */ ggml_backend_ponn_buffer_type_interface,
                /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_ponn_reg(), i),
                /* .context  = */
                 new ggml_backend_ponn_buffer_type_context{
                    i, "PONN" + std::to_string(i)},
            };
        }
        ggml_backend_ponn_buffer_type_initialized = true;
    }

    return &ggml_backend_ponn_buffer_types[device];
}

/**
 * @brief Computes the forward operation for a given tensor using PONN
 * operations.
 *
 * This function selects the appropriate PONN operation based on the type of
 * operation specified in the tensor and performs the computation.
 *
 * @param ctx The PONN context containing necessary resources and
 * configurations.
 * @param dst The destination tensor where the result of the computation will be
 * stored.
 * @return true if the computation was successful; false otherwise.
 */
static bool ggml_ponn_compute_forward(ggml_backend_ponn_context& ctx,
                                      struct ggml_tensor* dst) {
    GGML_PONN_TRACE();
    switch (dst->op) {
        case GGML_OP_REPEAT:
            ggml_ponn_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_ponn_get_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_ponn_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
            ggml_ponn_add(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_ponn_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_ponn_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_ponn_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_ponn_unary(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_ponn_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_ponn_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_ponn_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_ponn_upsample_nearest2d(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_ponn_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_ponn_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_ponn_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_ponn_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_ponn_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_ponn_mul_mat(ctx, dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            return false;
        case GGML_OP_SCALE:
            ggml_ponn_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_ponn_sqr(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_ponn_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_ponn_cpy(ctx, dst);
            break;
        case GGML_OP_CONT:
            ggml_ponn_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_ponn_diag_mask(ctx, dst, -INFINITY);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_ponn_soft_max(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_ponn_rope(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_ponn_im2col(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_ponn_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_ponn_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_ponn_argsort(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
}

// backend
/**
 * @brief Retrieves the name associated with the PONN backend.
 *
 * This function returns the name assigned to the PONN backend, which is stored
 * in the context of the provided backend structure.
 *
 * @param backend Pointer to the PONN backend structure.
 * @return A pointer to a constant string representing the backend name.
 */
static const char* ggml_backend_ponn_name(ggml_backend_t backend) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_context* ponn_ctx =
        (ggml_backend_ponn_context*)backend->context;
    return ponn_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the PONN backend.
 *
 * This function releases resources associated with the PONN backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the PONN backend structure to be freed.
 */
static void ggml_backend_ponn_free(ggml_backend_t backend) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_context* ponn_ctx =
        (ggml_backend_ponn_context*)backend->context;
    //GGML_PONN_NOT_SUPPORT();
#if 0
    PONN_CHECK(aclrtSynchronizeDevice());
    PONN_CHECK(aclrtResetDevice(ponn_ctx->device));

    // finalize when last backend freed.
    if (ponn_ctx->device == ggml_backend_ponn_get_device_count() - 1) {
        PONN_CHECK(aclFinalize());
    }
#endif
    delete ponn_ctx;
    delete backend;
}

/**
 * @brief Retrieves the default buffer type associated with the PONN backend.
 *
 * This function returns the buffer type specific to the device associated
 * with the PONN backend. It is used to allocate buffers for computations
 * performed by the backend.
 *
 * @param backend Pointer to the PONN backend structure.
 * @return Pointer to the buffer type structure for the PONN backend.
 */
static ggml_backend_buffer_type_t
ggml_backend_ponn_get_default_buffer_type(ggml_backend_t backend) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_context* ponn_ctx =
        (ggml_backend_ponn_context*)backend->context;

    return ggml_backend_ponn_buffer_type(ponn_ctx->device);
}

/**
 * @brief Sets tensor data asynchronously in the PONN backend.
 *
 * This function asynchronously sets tensor data in the PONN backend. Depending
 * on the tensor type, it may perform data transformations before copying data
 * to the device.
 *
 * @param backend Pointer to the PONN backend structure.
 * @param tensor Pointer to the tensor structure to set data for.
 * @param data Pointer to the host data to copy to the tensor.
 * @param offset Offset in bytes within the host data.
 * @param size Size of the data to copy in bytes.
 */
static void ggml_backend_ponn_set_tensor_async(ggml_backend_t backend,
                                                         ggml_tensor *tensor,
                                                         const void *data,
                                                         size_t offset,
                                                         size_t size) {
    GGML_PONN_TRACE();
    GGML_PONN_NOT_SUPPORT();
#if 0
    ggml_backend_ponn_context *ponn_ctx =
        (ggml_backend_ponn_context *)backend->context;
    if (!need_transform(tensor->type)) {
        PONN_CHECK(aclrtMemcpyAsync((char *)tensor->data + offset, size, data,
                                   size, ACL_MEMCPY_HOST_TO_DEVICE,
                                   ponn_ctx->stream()));
    } else {
        void *transform_buffer = malloc(size);
        ggml_backend_ponn_transform(tensor, data, transform_buffer);

#ifndef NDEBUG
        void *check_buffer = malloc(size);
        ggml_backend_ponn_transform_back(tensor, transform_buffer,
                                         check_buffer);
        GGML_ASSERT(memcmp(data, check_buffer, size));
        free(check_buffer);
#endif
        PONN_CHECK(aclrtMemcpyAsync(
            (char *)tensor->data + offset, size, transform_buffer, size,
            ACL_MEMCPY_HOST_TO_DEVICE, ponn_ctx->stream()));
        PONN_CHECK(aclrtSynchronizeStream(ponn_ctx->stream()));
        free(transform_buffer);
    }
#endif
}

static void ggml_backend_ponn_get_tensor_async(
    ggml_backend_t backend, const ggml_tensor *tensor, void *data,
    size_t offset, size_t size) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_context *ctx =
        (ggml_backend_ponn_context *)backend->context;
    ggml_backend_buffer_t buf =
        tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_ponn_buffer_type(ctx->device) &&
                "unsupported buffer type");
    //not async ...fix me
    //GGML_PONN_NOT_SUPPORT();
    ggml_ponn_set_device(ctx->device);
    if (!need_transform(tensor->type)) {
        ggml_tensor_extra_gpu *extra = (ggml_tensor_extra_gpu *)tensor->extra;
        std::vector<int> tensor_dims;
        ponn_utils_get_tensor_dims(tensor_dims, tensor->ne);
        ponnMemcpyEx(data, ponn_utils_get_data_type(tensor->type),  extra->handle, ponnGetInferenceDataType(),  size, tensor_dims, DEVICE_TO_HOST);
        // ponnMemcpy(data, 0, extra->handle, extra->offset + offset,  size, DEVICE_TO_HOST);
    } else {
        GGML_ASSERT(0);
    #if 0
        void *transform_buffer = malloc(size);
        PONN_CHECK(aclrtMemcpyAsync(
            transform_buffer, size, (char *)tensor->data + offset, size,
            ACL_MEMCPY_DEVICE_TO_HOST, ponn_ctx->stream()));
        PONN_CHECK(aclrtSynchronizeStream(ponn_ctx->stream()));
        ggml_backend_ponn_transform_back(tensor, transform_buffer, data);
        free(transform_buffer);
    #endif
    }
}

/**
 * @brief Asynchronously copies tensor data between PONN backends.
 *
 * This function copies tensor data asynchronously between two PONN backends. It
 * checks if both tensors reside in PONN buffers and whether the devices support
 * peer-to-peer access for direct copying. If not, it returns false.
 *
 * @param backend_src Pointer to the source PONN backend structure.
 * @param backend_dst Pointer to the destination PONN backend structure.
 * @param src Pointer to the source tensor to copy data from.
 * @param dst Pointer to the destination tensor to copy data to.
 * @return true if the copy operation succeeds, false otherwise.
 */
static bool ggml_backend_ponn_cpy_tensor_async(
    ggml_backend_t backend_src, ggml_backend_t backend_dst,
    const ggml_tensor* src, ggml_tensor* dst) {
    GGML_PONN_TRACE();
    GGML_ASSERT(ggml_backend_is_ponn(backend_src) ||
                ggml_backend_is_ponn(backend_dst));
    //GGML_PONN_NOT_SUPPORT();

    if (!ggml_backend_buffer_is_ponn(src->buffer) ||
        !ggml_backend_buffer_is_ponn(dst->buffer)) {
        return false;
    }
    ggml_backend_buffer_t buf_src =
        src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst =
        dst->view_src ? dst->view_src->buffer : dst->buffer;

    ggml_backend_ponn_context* ponn_ctx_src =
        (ggml_backend_ponn_context*)backend_src->context;
    ggml_backend_ponn_context* ponn_ctx_dst =
        (ggml_backend_ponn_context*)backend_dst->context;
    size_t copy_size = ggml_nbytes(dst);
    if (backend_src != backend_dst) {
        ggml_backend_ponn_buffer_context* buf_ctx_src =
            (ggml_backend_ponn_buffer_context*)buf_src->context;
        ggml_backend_ponn_buffer_context* buf_ctx_dst =
            (ggml_backend_ponn_buffer_context*)buf_dst->context;

        GGML_ASSERT(ponn_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(ponn_ctx_dst->device == buf_ctx_dst->device);
        size_t dst_offset = (char *)dst->data - (char *)buf_ctx_dst->dev_ptr;
        size_t src_offset = (char *)src->data - (char *)buf_ctx_src->dev_ptr;
        ponnMemcpy(buf_ctx_dst->dev_ptr, dst_offset, buf_ctx_src->dev_ptr, src_offset, copy_size, DEVICE_TO_DEVICE);
#if 0
        int32_t canAccessPeer = 0;
        PONN_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, ponn_ctx_src->device,
                                           ponn_ctx_dst->device));
        if (!canAccessPeer) {
            return false;
        }
        // need open both directions for memcpyasync between devices.
        ggml_ponn_set_device(ponn_ctx_dst->device);
        PONN_CHECK(aclrtDeviceEnablePeerAccess(ponn_ctx_src->device, 0));
        ggml_ponn_set_device(ponn_ctx_src->device);
        PONN_CHECK(aclrtDeviceEnablePeerAccess(ponn_ctx_dst->device, 0));

        PONN_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   ponn_ctx_src->stream()));

        //TODO: workaround for Event didn`t work here.
        aclrtSynchronizeStream(ponn_ctx_src->stream());
#endif
    } else {
#if 0
        // src and dst are on the same backend
        PONN_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   ponn_ctx_dst->stream()));
#endif
    }
    return true;
}

/**
 * @brief Synchronizes a PONN backend.
 *
 * This function synchronizes the specified PONN backend by waiting for all
 * operations in its associated stream to complete.
 *
 * @param backend Pointer to the PONN backend structure to synchronize.
 */
static void ggml_backend_ponn_synchronize(ggml_backend_t backend) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_context* ponn_ctx =
        (ggml_backend_ponn_context*)backend->context;

    ggml_ponn_set_device(ponn_ctx->device);
    ponnSyncStream(ponn_ctx->stream());
}

bool ponn_is_same_tensor(struct ggml_tensor_extra_gpu *t0, struct ggml_tensor_extra_gpu *t1){
    if (t0->buf_handle == t1->buf_handle && t0->buf_offset == t1->buf_offset) {
        return true;
    }
    return false;
}

/**
 * @brief Computes a computational graph using a PONN backend.
 *
 * This function computes the operations defined in the computational graph
 * using the specified PONN backend.
 *
 * @param backend Pointer to the PONN backend structure to use for computation.
 * @param cgraph Pointer to the computational graph structure containing nodes
 *               representing operations to be computed.
 * @return enum ggml_status Returns GGML_STATUS_SUCCESS if computation
 *         completes successfully, otherwise an appropriate error status.
 */
static enum ggml_status ggml_backend_ponn_graph_compute(
    ggml_backend_t backend, ggml_cgraph* cgraph) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_context* ctx =
        (ggml_backend_ponn_context*)backend->context;

    ggml_ponn_set_device(ctx->device);
#ifdef GGML_PONN_PROFILING_PRECISION
    GGML_LOG_INFO("PONN precision -------------------------------------------------------------------- \n");
#endif
#ifndef GGML_PONN_PROFILING
    //通过nodes[0]的RMS NORM算子序列长度，判断是prompt还是output
    cgraph->nodes[0]->ne[1] == 1 ? ponnSetEnableSync(false) : ponnSetEnableSync(ponnGetPromptSync());
#endif
    int fused_index = -1;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];

        if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
            GGML_LOG_INFO("%s: ponn graph compute aborted\n", __func__);
            return GGML_STATUS_ABORTED;
        }

        if (ggml_is_empty(node) || node->op == GGML_OP_NONE || i == fused_index) {
            continue;
        }
    #ifdef GGML_PONN_PROFILING
        auto st = std::chrono::system_clock::now();
    #endif
        ggml_tensor* next_node = cgraph->nodes[i+1];
        // to fix overflow in matmul op.
        if (node->op == GGML_OP_MUL_MAT && next_node && next_node->op == GGML_OP_SOFT_MAX) {
            struct ggml_tensor_extra_gpu *extra0 = (struct ggml_tensor_extra_gpu *)node->extra;
            struct ggml_tensor_extra_gpu *extra1 = (struct ggml_tensor_extra_gpu *)next_node->src[0]->extra;
            if (ponn_is_same_tensor(extra0, extra1) && next_node->op_params[0]) {
                node->op_params[1] = next_node->op_params[0];
            }
        }
#ifndef GGML_PONN_PROFILING_PRECISION
        ggml_tensor* next_5_node = cgraph->nodes[i+5];
        if (node->op == GGML_OP_MUL_MAT && next_node && next_node->op == GGML_OP_ADD) {
            struct ggml_tensor_extra_gpu *extra0 = (struct ggml_tensor_extra_gpu *)node->extra;
            struct ggml_tensor_extra_gpu *extra1 = (struct ggml_tensor_extra_gpu *)next_node->extra;
            if (ponn_is_same_tensor(extra0, extra1) && strstr(next_node->src[1]->name, "bias")) {
                node->src[2] = next_node->src[1];
                i++;
            }
        }
        else if(node->op == GGML_OP_MUL_MAT && next_5_node && next_5_node->op == GGML_OP_ADD) {
            struct ggml_tensor_extra_gpu *extra0 = (struct ggml_tensor_extra_gpu *)node->extra;
            struct ggml_tensor_extra_gpu *extra1 = (struct ggml_tensor_extra_gpu *)next_5_node->extra;
            if (ponn_is_same_tensor(extra0, extra1) && strstr(next_5_node->src[1]->name, "bias")) {
                node->src[2] = next_5_node->src[1];
                fused_index = i + 5;
            }
        }
        else if (node->op == GGML_OP_RMS_NORM && next_node && next_node->op == GGML_OP_MUL) {
            struct ggml_tensor_extra_gpu *extra0 = (struct ggml_tensor_extra_gpu *)node->extra;
            struct ggml_tensor_extra_gpu *extra1 = (struct ggml_tensor_extra_gpu *)next_node->extra;
            if (ponn_is_same_tensor(extra0, extra1) && strstr(next_node->src[1]->name, "norm.weig")) {
                node->src[1] = next_node->src[1];
                i++;
            }
        }
        else if(node->op == GGML_OP_MUL_MAT){
            if(next_node && next_node->op == GGML_OP_SCALE) {
                struct ggml_tensor_extra_gpu *extra0 = (struct ggml_tensor_extra_gpu *)node->extra;
                struct ggml_tensor_extra_gpu *extra1 = (struct ggml_tensor_extra_gpu *)next_node->extra;
                if (ponn_is_same_tensor(extra0, extra1) && next_node->op_params[0]) {
                    node->op_params[1] = next_node->op_params[0];
                    i++;
                    next_node = cgraph->nodes[i+1];
                    if(next_node && next_node->op == GGML_OP_ADD) {
                        extra1 = (struct ggml_tensor_extra_gpu *)next_node->src[1]->extra;
                        if (ponn_is_same_tensor(extra0, extra1)) {
                            node->src[2] = next_node->src[0];
                            next_node->extra =  extra0;
                            i++;
                        }
                    }
                }
            }
        }
#endif
        bool ok = ggml_ponn_compute_forward(*ctx, node);

        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__,
                    node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    #ifdef GGML_PONN_PROFILING_PRECISION
        GGML_LOG_INFO("node %4d ", i);
        GGML_LOG_INFO("name %20.20s  ", node->name);
        ponn_utils_tensor_precision_info(node);
        //ponn_utils_tensor_precision_info(node->src[0]);
        GGML_LOG_INFO("\n");
    #endif
    #ifdef GGML_PONN_PROFILING
        ctx->profiler[ggml_op_name(node->op)] += ponn_utils_get_span(st, std::chrono::system_clock::now());
    #endif
    }

#ifdef GGML_PONN_PROFILING_PRECISION
    GGML_LOG_INFO("PONN precision ==================================================================== \n");
#endif

#ifdef GGML_PONN_PROFILING
    ctx->print_profiling();
    ctx->clear_profiling();
#endif
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Checks if the PONN backend supports a specific operation.
 *
 * This function checks whether the specified operation is supported by the
 * PONN backend.
 *
 * @param backend Pointer to the PONN backend structure to check support for
 *                the operation.
 * @param op Pointer to the tensor representing the operation to check.
 * @return bool Returns true if the operation is supported by the backend,
 *              otherwise false.
 */
static bool ggml_backend_ponn_supports_op(ggml_backend_dev_t dev,
                                                    const ggml_tensor* op) {
    GGML_PONN_TRACE();
#if 0
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_UNARY:
        case GGML_OP_CONT:
        case GGML_OP_ROPE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_MUL_MAT:
        case GGML_OP_GET_ROWS:
            return true;
        default:
            return false;
    }
    return false;
#endif
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT: {
            switch (op->src[0]->type) {
                case GGML_TYPE_F16:
                case GGML_TYPE_F32:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_MUL_MAT_ID:
            return false;
        // embedding
        case GGML_OP_GET_ROWS: {
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_1:
                    return true;
                default:
                    return false;
            }
        } break;
        case GGML_OP_CPY: {
            switch (op->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_DUP:
        case GGML_OP_REPEAT:
        case GGML_OP_CONCAT:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
            return true;
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

/**
 * @brief Checks if the backend buffer type is associated with the PONN backend.
 *
 * This function checks whether the provided backend buffer type is associated
 * with the PONN backend based on the comparison of its name retrieval function
 * pointer.
 *
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the buffer type is associated with the PONN
 * backend, otherwise false.
 */
static bool ggml_backend_buft_is_ponn(ggml_backend_buffer_type_t buft) {
    GGML_PONN_TRACE();
    return buft->iface.get_name == ggml_backend_ponn_buffer_type_name;
}


/**
 * @brief Determines if a tensor operation should be offloaded to the PONN
 * backend.
 *
 * This function checks if a given tensor operation should be offloaded to the
 * PONN backend based on the operation type and the size of the tensor. It
 * returns true if the second dimension (ne[1]) of the tensor is greater than or
 * equal to the minimum batch size and the operation is not GGML_OP_GET_ROWS.
 *
 * @param backend Pointer to the PONN backend.
 * @param op Pointer to the tensor operation to check.
 * @return bool Returns true if the operation should be offloaded, otherwise
 * false.
 */
static bool ggml_backend_ponn_offload_op(ggml_backend_dev_t dev,
                                                   const ggml_tensor* op) {
    GGML_PONN_TRACE();
    const int min_batch_size = 32;
    GGML_UNUSED(dev);

    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;
}

/**
 * @brief Records an event on the PONN backend stream.
 *
 * This function records the given event on the PONN runtime stream associated
 * with the backend context.
 *
 * @param event Pointer to the event structure to be recorded.
 */
static void ggml_backend_ponn_event_record(ggml_backend_event_t event) {
    GGML_PONN_TRACE();
    GGML_PONN_NOT_SUPPORT();
#if 0
    ggml_backend_ponn_context* ponn_ctx =
        (ggml_backend_ponn_context*)event->backend->context;
    PONN_CHECK(aclrtRecordEvent((aclrtEvent)event->context, ponn_ctx->stream()));
#endif
}

/**
 * @brief Waits for a recorded event to complete on the PONN backend stream.
 *
 * This function makes the given backend wait for the event to complete on its
 * PONN runtime stream.
 *
 * @param backend Pointer to the backend structure.
 * @param event Pointer to the event structure that the backend needs to wait
 * for.
 */
static void ggml_backend_ponn_event_wait(ggml_backend_t backend,
                                         ggml_backend_event_t event) {
    GGML_PONN_TRACE();
    ggml_backend_ponn_context* ponn_ctx =
        (ggml_backend_ponn_context*)backend->context;

    GGML_PONN_NOT_SUPPORT();
#if 0
    if (ggml_backend_is_ponn(event->backend)) {
        PONN_CHECK(aclrtStreamWaitEvent(ponn_ctx->stream(),
                                       (aclrtEvent)event->context));
    } else {
        GGML_ABORT("fatal error");
    }
#endif
}

/**
 * @brief Synchronizes the given event on the PONN backend.
 *
 * This function waits for the specified event to complete on the PONN runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
static void ggml_backend_ponn_event_synchronize(ggml_backend_event_t event) {
    GGML_PONN_TRACE();
    GGML_PONN_NOT_SUPPORT();
#if 0
    PONN_CHECK(aclrtSynchronizeEvent((aclrtEvent)event->context));
#endif
}

/**
 * @brief Structure defining the interface for the PONN backend.
 *
 * This structure contains function pointers for various operations
 * supported by the PONN backend, including name retrieval, memory
 * management, tensor operations, synchronization, and event handling.
 */
static ggml_backend_i ggml_backend_ponn_interface = {
    /* .get_name                = */ ggml_backend_ponn_name,
    /* .free                    = */ ggml_backend_ponn_free,
//    /* .get_default_buffer_type = */ ggml_backend_ponn_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_ponn_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_ponn_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_ponn_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_ponn_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_ponn_graph_compute,
    ///* .supports_op             = */ ggml_backend_ponn_supports_op,
    ///* .supports_buft           = */ ggml_backend_ponn_supports_buft,
    ///* .offload_op              = */ ggml_backend_ponn_offload_op,
    ///* .event_record            = */ ggml_backend_ponn_event_record,
    ///* .event_wait              = */ ggml_backend_ponn_event_wait,
};

/**
 * @brief Return the hardcoded GUID for the PONN backend.
 *
 * This function returns a static GUID which uniquely identifies the PONN
 * backend.
 *
 * @return A pointer to the static GUID.
 */
static ggml_guid_t ggml_backend_ponn_guid() {
    GGML_PONN_TRACE();
    static ggml_guid guid = {0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                             0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x66};
    return &guid;
}

// backend device
struct ggml_backend_ponn_device_context {
    int device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_ponn_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_ponn_device_context * ctx = (ggml_backend_ponn_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char* ggml_backend_ponn_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_ponn_device_context * ctx = (ggml_backend_ponn_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_ponn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_ponn_device_context * ctx = (ggml_backend_ponn_device_context *)dev->context;
    ggml_backend_ponn_get_device_memory(ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_ponn_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_ponn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_ponn_device_get_name(dev);
    props->description = ggml_backend_ponn_device_get_description(dev);
    props->type        = ggml_backend_ponn_device_get_type(dev);
    ggml_backend_ponn_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_PONN_NO_PINNED") == nullptr;

    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
    };
}

static ggml_backend_t ggml_backend_ponn_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_ponn_device_context * ctx = (ggml_backend_ponn_device_context *)dev->context;
    return ggml_backend_ponn_init(ctx->device);
}


static ggml_backend_buffer_type_t ggml_backend_ponn_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_ponn_device_context * ctx = (ggml_backend_ponn_device_context *)dev->context;
    return ggml_backend_ponn_buffer_type(ctx->device);
}
#if 0
static ggml_backend_buffer_type_t ggml_backend_ponn_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_ponn_host_buffer_type();
}
#endif

/**
 * @brief Checks if the PONN backend supports a specific backend buffer type.
 *
 * This function determines whether the PONN backend supports the given backend
 * buffer type by comparing the device context of the backend and buffer type.
 * It returns true if the devices are same between the backend context and
 * buffer type context.
 *
 * @param backend Pointer to the PONN backend.
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the PONN backend supports the buffer type,
 *              otherwise false.
 */
static bool ggml_backend_ponn_supports_buft(
    ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_PONN_TRACE();
    if (ggml_backend_buft_is_ponn(buft)) {
        ggml_backend_ponn_device_context * ponn_ctx =
                        (ggml_backend_ponn_device_context *)dev->context;
        ggml_backend_ponn_buffer_type_context * buft_ctx =
                        (ggml_backend_ponn_buffer_type_context *)buft->context;
        return buft_ctx->device == ponn_ctx->device;
    }
    return false;
}

static const ggml_backend_device_i ggml_backend_ponn_device_interface = {
    /* .get_name                = */ ggml_backend_ponn_device_get_name,
    /* .get_description         = */ ggml_backend_ponn_device_get_description,
    /* .get_memory              = */ ggml_backend_ponn_device_get_memory,
    /* .get_type                = */ ggml_backend_ponn_device_get_type,
    /* .get_props               = */ ggml_backend_ponn_device_get_props,
    /* .init_backend            = */ ggml_backend_ponn_device_init,    // called for every card
    /* .get_buffer_type         = */ ggml_backend_ponn_device_get_buffer_type,
    /* .get_host_buffer_type    = */ NULL, //ggml_backend_ponn_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL, // not supported for CANN
    /* .supports_op             = */ ggml_backend_ponn_supports_op,
    /* .supports_buft           = */ ggml_backend_ponn_supports_buft,
    /* .offload_op              = */ ggml_backend_ponn_offload_op,
    /* .event_new               = */ NULL, //ggml_backend_ponn_device_event_new,
    /* .event_free              = */ NULL, //ggml_backend_ponn_device_event_free,
    /* .event_synchronize       = */ NULL, //ggml_backend_ponn_device_event_synchronize,
};

// backend reg
struct ggml_backend_ponn_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_ponn_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_PONN_NAME;
}

static size_t ggml_backend_ponn_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_ponn_reg_context * ctx = (ggml_backend_ponn_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_ponn_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_ponn_reg_context * ctx = (ggml_backend_ponn_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_ponn_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    // reserved for future use
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_set_abort_callback") == 0) {
        return (void *)ggml_backend_ponn_set_abort_callback;
    }

    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_ponn_reg_interface = {
    /* .get_name          = */ ggml_backend_ponn_reg_get_name,
    /* .get_device_count  = */ ggml_backend_ponn_reg_get_device_count,
    /* .get_device        = */ ggml_backend_ponn_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_ponn_reg_get_proc_address,
};

// backend registry, called only once for cann backend
ggml_backend_reg_t ggml_backend_ponn_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;
    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            //aclInit(nullptr);
            ggml_backend_ponn_reg_context * ctx = new ggml_backend_ponn_reg_context;

            for (int i = 0; i < ggml_ponn_info().device_count; i++) {
                ggml_backend_ponn_device_context* dev_ctx = new ggml_backend_ponn_device_context();
                dev_ctx->description = "GLF";//aclrtGetSocName();
                dev_ctx->device = i;
                dev_ctx->name = GGML_PONN_NAME + std::to_string(i);
                ggml_ponn_set_device(i);
                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_ponn_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_ponn_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_ponn_init(int32_t device) {
    GGML_PONN_TRACE();
    if (device < 0 || device >= ggml_backend_ponn_get_device_count()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_ponn_context* ctx = new ggml_backend_ponn_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t ponn_backend =
        new ggml_backend{/* .guid      = */ ggml_backend_ponn_guid(),
                         /* .interface = */ ggml_backend_ponn_interface,
                         /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_ponn_reg(), device),
                         /* .context   = */ ctx};

    return ponn_backend;
}

bool ggml_backend_is_ponn(ggml_backend_t backend) {
    GGML_PONN_TRACE();
    return backend != NULL &&
           ggml_guid_matches(backend->guid, ggml_backend_ponn_guid());
}

int32_t ggml_backend_ponn_get_device_count() {
    GGML_PONN_TRACE();
    return ggml_ponn_info().device_count;
}

void ggml_backend_ponn_get_device_description(
    int32_t device, char* description, size_t description_size) {
    GGML_PONN_TRACE();
    ggml_ponn_set_device(device);
    const char* name = ponnGetName();
    snprintf(description, description_size, "%s", name);
}

void ggml_backend_ponn_get_device_memory(int32_t device, size_t* free,
                                                   size_t* total) {
    GGML_PONN_TRACE();
    ggml_ponn_set_device(device);
    ponnGetMemInfo(free, total);
}

void ggml_backend_ponn_set_abort_callback(ggml_backend_t backend,
                                                    ggml_abort_callback abort_callback,
                                                    void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_ponn(backend));

    struct ggml_backend_ponn_context * ctx = (struct ggml_backend_ponn_context *)backend->context;

    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

// backend registry
/**
 * @brief Initializes a PONN backend based on the provided parameters.
 *
 * This function initializes a PONN backend using the device index and then
 * initializes the backend using `ggml_backend_ponn_init`.
 *
 * @param params Parameters for initialization (unused in this implementation).
 * @param user_data User data containing the device index to initialize the
 * backend.
 * @return ggml_backend_t The initialized PONN backend.
 */
static ggml_backend_t ggml_backend_reg_ponn_init(const char* params,
                                                           void* user_data) {
    GGML_PONN_TRACE();
    ggml_backend_t ponn_backend =
        ggml_backend_ponn_init((int)(intptr_t)user_data);
    return ponn_backend;

    GGML_UNUSED(params);
}

extern "C" int ggml_backend_ponn_reg_devices();

/**
 * @brief Registers PONN (GLF) devices as backend options.
 *
 * This function initializes PONN, retrieves the number of available PONN
 * devices, and registers each device as a backend option using
 * `ggml_backend_register`. Each device is given a unique name based on
 * `GGML_PONN_NAME` followed by its index.
 *
 * @return int The number of PONN devices registered.
 */
int ggml_backend_ponn_reg_devices() {
    uint32_t device_count = ggml_backend_ponn_get_device_count();
    GGML_PONN_TRACE();
    // initialization
    for (uint32_t i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "PONN%d", i);
        //ggml_backend_register(name, ggml_backend_reg_ponn_init,
        //                      ggml_backend_ponn_buffer_type(i),
        //                      (void*)(intptr_t)i);
    }
    return device_count;
}
GGML_BACKEND_DL_IMPL(ggml_backend_ponn_reg)
