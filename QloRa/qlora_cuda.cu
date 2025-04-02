#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Dequantization: Convert int8 to fp16 using scale
__device__ __forceinline__ half dequantize_int8(int8_t q_val, float scale) {
    return __float2half(q_val * scale);
}

// CUDA Kernel for Low-Rank Adaptation (QLoRA)
__global__ void lora_matmul_int8(
    const int8_t *B, const int8_t *A, half *W, 
    const float scaleB, const float scaleA, 
    int d, int r, int k) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < d && col < k) {
        half sum = __float2half(0.0f);

        // Compute W = B * A
        for (int i = 0; i < r; i++) {
            half b_val = dequantize_int8(B[row * r + i], scaleB);
            half a_val = dequantize_int8(A[i * k + col], scaleA);
            sum = __hadd(sum, __hmul(b_val, a_val));
        }

        W[row * k + col] = sum;
    }
}

// Host function to launch kernel
void lora_matmul(
    torch::Tensor B, torch::Tensor A, torch::Tensor W, 
    float scaleB, float scaleA) {

    int d = B.size(0);
    int r = B.size(1);
    int k = A.size(1);

    dim3 blockSize(16, 16);
    dim3 gridSize((k + 15) / 16, (d + 15) / 16);

    lora_matmul_int8<<<gridSize, blockSize>>>(
        B.data_ptr<int8_t>(), A.data_ptr<int8_t>(), W.data_ptr<half>(),
        scaleB, scaleA, d, r, k);
}

// Register function for PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lora_matmul", &lora_matmul, "QLoRA Matrix Multiplication (CUDA)");
}
