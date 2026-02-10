#ifndef PONN_OPS_H
#define PONN_OPS_H

#include "common.h"

/**
 * @brief   Repeats a ggml tensor along each dimension to match the dimensions
 *          of another tensor.
 *
 * @details This function repeats the elements of a source ggml tensor along
 *          each dimension to create a destination tensor with the specified
 *          dimensions. The operation is performed using the ACL backend and
 *          executed asynchronously on the device.
 *
 * @param   ctx The PONN context used for operations.
 * @param   dst The ggml tensor representing the destination, which op is
 *              GGML_OP_REPEAT and specifies the desired dimensions.
 */
void ggml_ponn_repeat(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Adds two ggml tensors using the PONN backend.
 *
 * @details This function performs an element-wise addition of two tensors. In
 *          case the tensors do not have the same shape, one or both tensors
 *          will be broadcasted to match the shape of the other before the
 *          addition is performed.The formula for the operation is given by:
 *          \f[
 *              \text{dst} = \text{acl_src0} + \alpha \cdot \text{acl_src1}
 *          \f]
 *
 * @param ctx The PONN context used for operations.
 * @param dst The ggml tensor representing the destination, result of the
 *            addition is stored at dst->data, and dst->op is `GGML_OP_ADD`
 */
void ggml_ponn_add(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Applies the Leaky ReLU activation function to a tensor using the PONN
 *          backend.
 *
 * @details This function computes the Leaky ReLU activation for each element of
 *          the input tensor. The Leaky ReLU function allows a small gradient
 *          when the unit is not active (i.e., when the input is negative). The
 *          Leaky ReLU function is defined as:
 *          \f[
 *              \text{dst} = \max(0, src) + \text{negativeSlope} \cdot \min(0,
 *               src)
 *          \f]
 *          `negativeSlope` is in dst->params.
 *
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the result of the Leaky ReLU
 *            activation is stored, which op is `GGML_OP_LEAKY_RELU`
 */
void ggml_ponn_leaky_relu(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief    Concatenates multiple tensors along a specified dimension using the
 *           PONN backend.
 *
 * @param ctx        The PONN context used for operations.
 * @param tensorList A pointer to the list of tensors to be concatenated.
 * @param dst        The destination tensor where the result of the
 *                   concatenation is stored. dst->op is `GGML_OP_CONCAT`.
 * @param concat_dim The dimension along which the tensors are concatenated.
 *
 * @attention tensorList length should be 2 and the dimension using for concat
 *            default to 1.
 */
void ggml_ponn_concat(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Generates a sequence of evenly spaced values within a specified
 *          interval for a ggml tensor using the PONN backend.
 *
 * @details This function creates a sequence of numbers over a specified i
 *          nterval, starting from `start`, ending before `stop`, and
 *          incrementing by `step`. The sequence is stored in the destination
 *          tensor `dst`.
 *
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the generated sequence will be stored.
 *            `start`, 'stop' and 'step' are in dst->op_params and dst->op is
 *            `GGML_OP_ARANGE`.
 */
void ggml_ponn_arange(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Computes the square of the elements of a ggml tensor using the PONN
 *          backend.
 * @details The function sets the second source tensor of the destination
 *          tensor `dst` to be equal to the first source tensor. This is
 *          effectively squaring the elements since the multiplication becomes
 *          `element * element`.
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the squared values will be stored，
 *            which dst->op is `GGML_OP_SQR`.
 */
void ggml_ponn_sqr(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Applies a clamp operation to the elements of a ggml tensor using the
 *          PONN backend.
 *
 * @details This function clamps the elements of the input tensor `src` to a
 *          specified range defined by `min` and `max` values. The result is
 *          stored in the destination tensor `dst`. The operation is defined as:
 *          \f[
 *              y = \max(\min(x, max\_value), min\_value)
 *           \f]
 *          where `x` is an element of the input tensor, and `y` is the
 *          corresponding element in the output tensor.
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the clamped values will be stored.
 *            dst->op is `GGML_OP_CLAMP`, `min` and `max` value is in dst->params.
 */
void ggml_ponn_clamp(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Scales the elements of a ggml tensor by a constant factor using the
 *          PONN backend.
 *
 * @details This function multiplies each element of the input tensor `src` by
 *          a scaling factor `scale`, storing the result in the destination
 *          tensor `dst`. The operation is defined as:
 *          \f[
 *             dst = src \times scale
 *          \f]
 *
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the scaled values will be stored.
 *            dst->op is `GGML_OP_SCALE` and `scale` value is in dst->params.
 */
void ggml_ponn_scale(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Sorts the elements of a ggml tensor and returns the indices that
 *          would sort the tensor using the PONN backend.
 *
 * @details This function performs an argsort operation on the input tensor
 *          `src`. It sorts the elements of `src` in either ascending or
 *          descending order, depending on the `GGML_SORT_ORDER_DESC`,
 *          and returns the indices that would sort the original tensor.
 *
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the sorted indices will be stored.
 *            dst->op is `GGML_OP_ARGSORT`.
 */
void ggml_ponn_argsort(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Computes the Layer Normalization for a ggml tensor using the PONN
 *          backend.
 *
 * @details This function applies the Layer Normalization operation on the
 *          input tensor `src` and stores the result in the destination tensor
 *          `dst`. Layer Normalization normalizes the features at each sample in
 *          a mini-batch independently. It is commonly used in neural networks
 *          to normalize the activations of a layer by adjusting and scaling
 *          the outputs.
 *          The operation is defined as:
 *          \f[
 *              \text { out }=\frac{x-\mathrm{E}[x]}{\sqrt{\text{Var}[x]+eps}}
 *          \f]
 *          `Var` defaults dst->ne[0]. `eps` is in dst->params.
 *
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the normalized values will be stored.
 * @attention `Var` defaults to dst->ne[0].
 */
void ggml_ponn_norm(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief  Computes the Group Normalization for a ggml tensor using the PONN
 *         backend.
 *
 * @brief  This function applies the Group Normalization operation on the input
 *         tensor `src` and stores the result in the destination tensor `dst`.
 *         Group Normalization divides the channels into groups and normalizes
 *         the features within each group across spatial locations.
 *         It is commonly used in convolutional neural networks to improve
 *         training stability and performance.
 *         The operation is defined as:
 *         \f[
 *             \text { out }=\frac{x-\mathrm{E}[x]}{\sqrt{\text{Var}[x]+eps}}
 *         \f]
 *
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the normalized values will be stored.
 *            `n_groups` is in dst->params, which split C channel to `n_groups`.
 *            dst->op is `GGML_OP_GROUP_NORM`.
 *
 * @attention eps defaults to 1e-6f.
 */
void ggml_ponn_group_norm(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Computes the accumulation of tensors using the PONN backend.
 *
 * @details This function performs an accumulation operation on two tensors.
 *          Depending on the `inplace` flag, it either updates the destination
 *          tensor `dst` in place by adding `alpha * src1` to it, or it creates
 *          a new tensor as the result of `src0 + alpha * src1` and stores it in
 *          `dst`.
 *          The operation is defined as:
 *          \f[
 *               dst = src0 + alpha \times src1
 *          \f]
 *          if `inplace` is `true`, `src0` is equal to 'dst'.
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the accumulated values will be stored.
 *            `inplace` is in dst->params, and dst->op is `GGML_OP_ACC`.
 */
void ggml_ponn_acc(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Computes the sum of elements along the last dimension of a ggml tensor
 *          using the PONN backend.
 *
 * @details This function performs a reduction sum operation along the last
 *          dimension of the input tensor `src`. The result of the sum is stored
 *          in the destination tensor `dst`.
 *
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the reduced values will be stored。
 *            dst->op is `GGML_OP_SUM_ROWS`.
 *
 * @attention `reduce_dims` defaults to 3, which means the last dimension.
 */
void ggml_ponn_sum_rows(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Upsamples a ggml tensor using nearest neighbor interpolation using
 *          the PONN backend.
 *
 * @details This function performs upsampling of the input tensor `src` using
 *          nearest neighbor interpolation. The upsampling is applied to the
 *          height and width dimensions (last two dimensions) of the tensor. The
 *          result is stored in the destination tensor `dst`, which must have
 *          the appropriate dimensions for the upsampled output.
 *
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the upsampled values will be stored.
 *            dst->op is `GGML_OP_UPSCALE`.
 */
void ggml_ponn_upsample_nearest2d(ggml_backend_ponn_context& ctx,
                                  ggml_tensor* dst);

/**
 * @brief   Pads a ggml tensor to match the dimensions of the destination tensor
 *          using the PONN backend.
 *
 * @details This function pads the input tensor `src` so that it matches the
 *          dimensions of the destination tensor `dst`. The amount of padding
 *          is calculated based on the difference in sizes between `src` and
 *          `dst` along each dimension. The padded tensor is stored in `dst`.
 *
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor, which specifies the target dimensions for
 *            padding. dst->op is `GGML_OP_PAD`.
 */
void ggml_ponn_pad(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Executes a 2D pooling operation on a ggml tensor using the PONN
 *          backend.
 *
 * @details This function dispatches the execution of a 2D pooling operation on
 *          the input tensor `dst`. The type of pooling (average or max) is
 *          determined by the `op` parameter, which is read from the operation
 *          parameters of `dst`. The function supports average pooling
 *          (`GGML_OP_POOL_AVG`) and max pooling (`GGML_OP_POOL_MAX`). If an
 *          invalid operation is encountered, the function asserts a failure.
 *
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor on which the pooling operation is to be
 *            performed. dst->op is `GGML_OP_POOL_2D`.
 */
void ggml_ponn_pool2d(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Duplicates a ggml tensor using the PONN backend.
 *
 * @details This function duplicates the contents of the source tensor `src` to
 *          the destination tensor `dst`. The function supports various tensor
 *          types and configurations, including handling of extra data, type
 *          conversions, and special cases for contiguous and non-contiguous
 *          tensors.
 *
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the duplicated data will be stored.
 *            dst->op is `GGML_OP_DUP`
 *
 * @attention Only support Fp16/FP32. Not support when src and dst have
 *            different shape and dst is no-contiguous.
 * @note:     This func need to simplify.
 */
void ggml_ponn_dup(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Computes the Root Mean Square (RMS) normalization of a ggml tensor
 *          using the PONN backend.
 *
 * @details This function applies RMS normalization to the input tensor `src`
 *          and stores the result in the destination tensor `dst`. RMS
 *          normalization involves computing the root mean square of the input
 *          tensor along a specified dimension and then dividing each element of
 *          the tensor by this value, adjusted by a small epsilon value to
 *          prevent division by zero.
 *          The operation is defined as:
 *          \f[
 *               \text{RmsNorm}\left(x_i\right)=\frac{x_i}{\text{Rms}(\mathbf{x})} g_i,
 *               \quad \text { where } \text{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+e p s}
 *          \f]
 *          `eps` is in dst->op_params.
 * @param ctx The PONN context used for operations.
 * @param dst The destination tensor where the normalized values will be stored.
 *            dst->op is `GGML_OP_RMS_NORM`.
 */
void ggml_ponn_rms_norm(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Applies a diagonal mask to the tensor with a specified value.
 *
 * @details This function creates a mask tensor filled with ones, then applies
 *          an upper triangular and lower triangular operation to it based on
 *          the number of past elements specified. Afterward, it adds the masked
 *          tensor to the destination tensor in-place.
 *
 * @param ctx The backend PONN context used for operations.
 * @param dst The destination tensor where the result will be stored. dst->op is
 *            `GGML_OP_DIAG_MASK`
 * @param value The value to use for masking.
 */
void ggml_ponn_diag_mask(ggml_backend_ponn_context& ctx, ggml_tensor* dst, float value);

/**
 * @brief   Performs an image-to-column transformation on the input tensor.
 *
 * @details This function takes an input tensor and applies an image-to-column
 *          operation, converting spatial dimensions into column-like
 *          structures suitable for convolutional operations. It supports both
 *          half-precision (F16) and single-precision (F32) floating-point data
 *          types.
 *
 * @param ctx The backend PONN context for executing operations.
 * @param dst The destination tensor that stores the result of the operation.
 *            dst->op is `GGML_OP_IM2COL`.
 */
void ggml_ponn_im2col(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Computes time step embeddings using sine and cosine functions.
 *
 * @details This function calculates time step embeddings by applying sine and
 *          cosine transformations to a given input tensor, which is typically
 *          used in temporal models like diffusion models or transformers to
 *          encode time information effectively.
 *
 * @param ctx The backend PONN context for executing operations.
 * @param dst The destination tensor where the result of the embedding operation
 *            will be stored. dst->op is `GGML_OP_TIMESTEP_EMBEDDING`.
 */
void ggml_ponn_timestep_embedding(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

// @see ggml_ponn_dup.
void ggml_ponn_cpy(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Computes the soft max activation with optional masking.
 *
 * @details This function computes the soft max activation over the input tensor,
 *          optionally applying a mask and scaling factor. It supports both FP16
 *          and FP32 data types and can handle masking by broadcasting the mask
 *          across rows if necessary.
 *          The function performs the following steps:
 *          1. Multiplies the input tensor by a scale factor.
 *          2. Optionally casts the mask tensor to FP32 if it is in FP16 format.
 *          3. Broadcasts the mask tensor if its dimensions do not match the
 *             input tensor's dimensions.
 *          4. Adds the mask to the scaled input tensor.
 *          5. Applies the softmax activation function along the specified
 *             dimension.
 *
 * @param ctx The backend PONN context for executing operations.
 * @param dst The destination tensor where the result will be stored. dst->op is
 *            `GGML_OP_SOFTMAX`.
 */
void ggml_ponn_soft_max(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Extracts specific rows from a tensor based on indices.
 *
 * @details This function retrieves rows from a source tensor src0 according to
 *          the indices provided in another tensor src1 and stores the result in
 *          a destination tensor (\p dst). It supports different data types
 *          including F32, F16, Q4_0, and Q8_0.
 *
 * @param ctx The backend PONN context for executing operations.
 * @param dst The destination tensor where the extracted rows will be stored.
 *            dst->op is `GGML_OP_GET_ROWS`.
 */
void ggml_ponn_get_rows(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief   Executes matrix multiplication for the given tensor.
 *
 * @details This function performs matrix multiplication on the source tensors
 *          associated with the destination tensor. It supports matrix
 *          multiplication F32, F16, and Q8_0.
 *
 * @param ctx The backend PONN context for executing operations.
 * @param dst The destination tensor for storing the result of the matrix
 *            multiplication. dst->op is `GGML_OP_MUL_MAT`.
 */
void ggml_ponn_mul_mat(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

/**
 * @brief Applies Rotary Positional Embedding (RoPE) to the input tensor.
 *
 * @details This function implements the RoPE mechanism, which is a method to
 *          encode positional information into sequence data, particularly
 *          useful in transformer models. It supports both F32 and F16 data
 *          types.
 *
 * @param ctx The backend PONN context for executing operations.
 * @param dst The destination tensor where the RoPE-transformed data will be
 *            stored. dst->op is `GGML_OP_ROPE`.
 *
 * @note The function currently does not support cases where the n_dims is less
 *       than the input tensor's first dimension.
 * @note The function currently does not support cases where the freq_factors is
 *       not NULL.
 * @note The function currently does not support cases where the ext_factor is
 *       not equal 0.
 * @note The function currently does not support cases where the freq_scale is
 *       not equal 1.
 */
void ggml_ponn_rope(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

void ggml_ponn_mul(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

void ggml_ponn_div(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

//Unary functions
void ggml_ponn_unary(ggml_backend_ponn_context& ctx, ggml_tensor* dst);

void ggml_ponn_fallback(ggml_tensor * tensor);

#endif  // PONN_OPS_H
