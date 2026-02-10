#ifndef PONN_COMMON_H
#define PONN_COMMON_H

#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include "ggml-impl.h"
#include "ponn.h"
#include "ggml-ponn.h"
#include "utils.h"

#define MATRIX_ROW_PADDING 256
#define GGML_PONN_MAX_STREAMS 8

/**
 * @brief Handles PONN-related errors by printing an error message and
 *        terminating the program.
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number at which the error occurred.
 * @param msg The error message.
 */
[[noreturn]] void ggml_ponn_error(const char* stmt, const char* func,
                                  const char* file, int line, const char* msg);

/**
 * @brief Checks the result of a PONN function call and invokes the error
 *        handler if the call fails.
 * @param stmt The PONN function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#define PONN_CHECK_GEN(stmt, success, error_fn)                                \
    do {                                                                      \
        int err_code = (stmt);                                                \
        if (err_code != (success)) {                                          \
            ggml_ponn_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } while (0);

#define PONN_CHECK(stmt) PONN_CHECK_GEN(stmt, 0, 0) //aclGetRecentErrMsg)

/**
 * @brief Contains information about PONN devices.
 */
struct ggml_ponn_device_info {
    /**
     * @brief Number of PONN devices available.
     */
    int32_t device_count;

    /**
     * @brief Information about a single PONN device.
     */
    struct ponn_device_info {
        int cc;                 /**< Compute capability.                   */
        size_t smpb;            /**< Maximum shared memory per block.      */
        bool vmm;               /**< Virtual memory support.               */
        size_t vmm_granularity; /**< Granularity of virtual memory.        */
        size_t total_vram;      /**< Total video RAM available on the device. */
    };

    ponn_device_info devices[GGML_PONN_MAX_DEVICES] =
        {}; /**< Array of PONN device information. */
};

const ggml_ponn_device_info& ggml_ponn_info();

void ggml_ponn_set_device(int32_t device);
int32_t ggml_ponn_get_device();

/**
 * @brief Context for managing PONN backend operations.
 */
struct ggml_backend_ponn_context {
    int32_t device;                  /**< Device ID. */
    std::string name;                /**< Name of the device. */
#ifdef GGML_PONN_PROFILING
    std::map <std::string, float> profiler;
#endif
    void * copy_event = nullptr; /**< Event for managing copy operations. */
    void * streams[GGML_PONN_MAX_STREAMS] = {
        nullptr}; /**< Array of streams for the device. */

    PONN_MEM_H one_tensor = nullptr;
    PONN_MEM_H ponn_sin = nullptr;
    PONN_MEM_H ponn_cos = nullptr;
    size_t one_tensor_size = 0;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_ponn_context(int device)
        : device(device), name("PONN" + std::to_string(device)) {}

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_ponn_context() {
        if (copy_event != nullptr) {
            ponnDestroyEvent(copy_event);
        }
        for (int i = 0; i < GGML_PONN_MAX_STREAMS; ++i) {
            if (streams[i] != nullptr) {
                ponnDestroyStream(streams[i]);
            }
        }
        if (one_tensor) {
            ponnFreeBuf(one_tensor);
            one_tensor = nullptr;
            one_tensor_size = 0;
        }
        if (ponn_sin) {
            ponnFreeBuf(ponn_sin);
            ponn_sin = nullptr;
        }
        if (ponn_cos) {
            ponnFreeBuf(ponn_cos);
            ponn_cos = nullptr;
        }
        ponnDeinit();
    }
    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    void* stream(int stream) {
        if (streams[stream] == nullptr) {
            ggml_ponn_set_device(device);
            ponnCreateStream(&streams[stream]);
        }
        return streams[stream];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    void* stream() { return stream(0); }

#ifdef GGML_PONN_PROFILING
    void print_profiling() {
        std::vector<std::pair<std::string,float>> v(profiler.begin(), profiler.end());
        std::sort(v.begin(), v.end(), ponn_utils_cmp);
        double total = 0;
        GGML_LOG_INFO("\nprofiling info:-------------------- \n");
        for(size_t i = 0; i < v.size(); i++){
            total += v[i].second;
        }
        GGML_LOG_INFO("total %f s \n", total);
        for(size_t i = 0; i < v.size(); i++){
            if (total) {
                GGML_LOG_INFO("%10.10s spend %6.4f s, %5.3f% \n", v[i].first.c_str(), v[i].second, v[i].second * 100 / total);
            } else {
                GGML_LOG_INFO("%10.10s spend %6.4f s \n", v[i].first.c_str(), v[i].second);
            }
        }
        ponnPrintProfiling();
    }
    void clear_profiling() {
        profiler.clear();
        ponnClearProfiling();
    }
#endif
    PONN_MEM_H get_one_tensor(size_t size, PONN_DATA_TYPE_E ponn_datatype) {
        if (size > one_tensor_size) {
            if (one_tensor) {
                ponnFreeBuf(one_tensor);
            }
            one_tensor = ponnMallocBuf(size);
            one_tensor_size = size;
            if(ponn_datatype == PONN_DATA_FLOAT){
                float value = 1.0f;
                ponnMemset(one_tensor, 0, size, &value, sizeof(value));
            }
            else if(ponn_datatype == PONN_DATA_HALF){
                ggml_fp16_t value = ggml_fp32_to_fp16(1.0f);
                ponnMemset(one_tensor, 0, size, &value, sizeof(value));
            }

        }
        return one_tensor;
    }
};

struct ggml_tensor_extra_gpu {
    PONN_MEM_H handle;
    uint64_t offset;
    PONN_MEM_H buf_handle;
    uint64_t buf_offset;
    std::vector <PONN_MEM_H> extraPonnData;//{weight, scale, mins， bias}
};
bool ponn_is_same_tensor(struct ggml_tensor_extra_gpu *t0, struct ggml_tensor_extra_gpu *t1);

#endif  // PONN_COMMON_H
