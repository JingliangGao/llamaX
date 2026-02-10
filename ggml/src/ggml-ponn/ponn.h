#ifndef PONN_PONN_H
#define PONN_PONN_H

#include <chrono>

#ifdef  __cplusplus
extern "C" {
#endif

#define PONN_SEQ_LEN_ALIGN 1
#define PONN_SEQ_LEN_ALIGN_SIZE 32

typedef enum
{
    PONN_DATA_NONE = -1,
    PONN_DATA_FLOAT = 0,
    PONN_DATA_HALF  = 1,
    PONN_DATA_S32   = 2,
    PONN_DATA_S16   = 3,
    PONN_DATA_S8    = 4,
    PONN_DATA_U8    = 5,
    PONN_DATA_S64   = 6,
    PONN_DATA_INT4 = 7,
    PONN_DATA_QINT4_0,
    PONN_DATA_QINT4_1,
    PONN_ZXMASK_32BITS,
    PONN_DATA_LLAMA_NONE,
    PONN_DATA_LLAMA_Q6_K,
    PONN_DATA_LLAMA_Q4_1,
    PONN_DATA_LLAMA_Q4_K,
} PONN_DATA_TYPE_E;

typedef void * PONN_MEM_H;
typedef void * PONN_STREAM_H;

void ponnOclSeqLenAlign(std::vector<float> &ids);

void ponnInit(void);
void ponnDeinit(void);
void ponnPrintProfiling(void);
void ponnClearProfiling(void);
char * ponnGetName();
void ponnSetDataLayout(const int dataLayout);
bool ponnGetFoldStatus();
void ponnGetMemInfo(size_t *free, size_t *total);

PONN_DATA_TYPE_E ponnGetInferenceDataType();
void ponnSetInferenceDataType(PONN_DATA_TYPE_E dtype);
PONN_STREAM_H ponnGetStream();
void ponnSync(PONN_STREAM_H stream);

void ponnGetDeviceCount(int32_t* devCount);
void ponnGetDevice(int* gpu_id);
void ponnSetDevice(int gpu_id);

void ponnSyncStream(PONN_STREAM_H stream);
void ponnFlushStream(PONN_STREAM_H stream);
void ponnCreateStream(PONN_STREAM_H *stream);
void ponnDestroyStream(PONN_STREAM_H stream);

void ponnSetEnableSync(bool enable);
bool ponnGetEnableSync();
void ponnSetPromptSync(bool enable);
bool ponnGetPromptSync();

void ponnCreateEvent(void **event);
void ponnDestroyEvent(void *event);

#define PONN_BUFFER_ALIGNMENT 256

void ponnMallocBigBuffer(size_t size);
void ponnClearBuffer();
PONN_MEM_H ponnMallocBuf(size_t size);
void ponnFreeBuf(PONN_MEM_H ret);
void ponnFree(PONN_MEM_H ret);
PONN_MEM_H ponnMallocSubBuf(PONN_MEM_H handle, size_t offset, size_t size);
void ponnFreeSubBuf(PONN_MEM_H handle);
void * ponnGetBase(PONN_MEM_H handle);
void * ponnDirectMalloc(size_t size);
void ponnDirectFree(PONN_MEM_H ret);

void ponnMemset(PONN_MEM_H buf, size_t offset, size_t size, void *value, size_t valueSize);

typedef enum {
    HOST_TO_DEVICE=0,
    DEVICE_TO_HOST,
    HOST_TO_HOST,
    DEVICE_TO_DEVICE
} PONN_MEMCPY_KIND;


void ponnMemcpy(void *dst, size_t dst_offset, const void *src, size_t src_offset,
                size_t size, PONN_MEMCPY_KIND kind);
void ponnMemcpyEx(void *dst, PONN_DATA_TYPE_E dst_type, const void *src, PONN_DATA_TYPE_E src_type,
                size_t size, std::vector<int>&dims, PONN_MEMCPY_KIND kind);
void ponnMemcpyNoContiguous(PONN_MEM_H input, PONN_MEM_H output,
                            std::vector<int> &inputDims, std::vector<int> &inputStrides,
                            std::vector<int> &outputDims, std::vector<int> &outputStrides,
                            PONN_DATA_TYPE_E inputType, PONN_DATA_TYPE_E outputType,
                            size_t input_offset, size_t output_offset);

void ponnAdd(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                std::vector<int> &input0Dims, std::vector<int> &input1Dims, std::vector<int> &outputDims);
void ponnMul(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                std::vector<int> &input0Dims, std::vector<int> &input1Dims, std::vector<int> &outputDims);
void ponnScale(PONN_MEM_H input, PONN_MEM_H output, std::vector<int>& dims, float scale);
void ponnPermute(PONN_MEM_H input, PONN_MEM_H output, PONN_DATA_TYPE_E dataType,
                std::vector<int> &dst_dims, const std::vector<int> &axis);

void ponnMulMatFp(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                PONN_DATA_TYPE_E input0Type, PONN_DATA_TYPE_E input1Type, PONN_DATA_TYPE_E outputType,
                int input0Spatial, int input1Spatial, int outputSpatial,
                int batch, int n, int m, int k, int group, bool v_cache, float alpha);
void ponnMulMatFpBias(PONN_MEM_H input0, PONN_MEM_H input1,  PONN_MEM_H input2, PONN_MEM_H output,
                PONN_DATA_TYPE_E input0Type, PONN_DATA_TYPE_E input1Type,
                PONN_DATA_TYPE_E input2Type, PONN_DATA_TYPE_E outputType,
                int input0Spatial, int input1Spatial, int input2Spatial, int outputSpatial,
                int batch, int n, int m, int k, int group, float alpha);
void ponnMulMatQuant(PONN_MEM_H input, PONN_MEM_H weight, PONN_MEM_H output,
                        PONN_MEM_H scales, PONN_MEM_H mins, PONN_MEM_H bias,
                         int n, int m, int k, int block_size, PONN_DATA_TYPE_E quant_type);

void ponnSoftmax(PONN_MEM_H input, PONN_MEM_H output, std::vector<int>& dims);
void ponnSilu(PONN_MEM_H input, PONN_MEM_H output, std::vector<int> &dims);
void ponnGelu(PONN_MEM_H input, PONN_MEM_H output, std::vector<int> &dims);

void ponnRmsNorm(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                std::vector<int> &input0Dims, std::vector<int> &input1Dims,
                std::vector<int> &outputDims, float eps);

void ponnAttentionMask(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output, std::vector<int>& dims);

void ponnGetRows(PONN_MEM_H input0, PONN_MEM_H input1, PONN_MEM_H output,
                    std::vector<int> &input0Dims, std::vector<int> &input0Strides,
                    std::vector<int> &input1Dims, std::vector<int> &input1Strides,
                    std::vector<int> &outputDims, std::vector<int> &outputStrides,
                    PONN_DATA_TYPE_E inputType);

void ponnRope(PONN_MEM_H input, PONN_MEM_H posId, PONN_MEM_H sin,
                PONN_MEM_H cos, PONN_MEM_H output,
                std::vector<int> &inputDims, std::vector<int> &posIdDims,
                std::vector<int> &sinDims, std::vector<int> &cosDims,
                std::vector<int> &outputDims, const int rotary_dim,
                const bool is_neox);
#ifdef  __cplusplus
}
#endif
#endif  // PONN_PONN
