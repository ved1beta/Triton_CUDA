// Error checking macro
#define CUDA_CHECK(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(error)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

// Memory allocation helper
template <typename T>
T* allocateDeviceMemory(size_t size) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
    return ptr;
}

// Memory copy helper
template <typename T>
void copyToDevice(T* dst, const T* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice));
}