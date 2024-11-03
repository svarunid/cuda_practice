#include <cuda.h>

#include <iostream>

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " in file " << file
                  << " at line " << line << std::endl;
        exit(code);
    }
}

inline void cudaCheckError(cudaError_t code, const char *file = __FILE__, int line = __LINE__) {
    gpuAssert(code, file, line);
}

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

__global__ void vecAddKernel(float *a, float *b, float *c, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

void vecAdd(float *A, float *B, float *C, int n) {
    float *A_d, *B_d, *C_d;
    size_t size = n * sizeof(float);

    cudaCheckError(cudaMalloc((void **)&A_d, size));
    cudaCheckError(cudaMalloc((void **)&B_d, size));
    cudaCheckError(cudaMalloc((void **)&C_d, size));

    cudaCheckError(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice));

    const unsigned int numThreads = 256;
    unsigned int numBlocks = cdiv(n, numThreads);
    vecAddKernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d, n);

    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost));

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void initializeVector(float *matrix, int N) {
    for (int i = 0; i < N; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void printVector(const float *matrix, int N) {
    for (int i = 0; i < N; ++i) {
        std::cout << matrix[i] << " ";
    }
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    const int N = 1000;
    float A[N];
    float B[N];
    float C[N];

    initializeVector(A, N);
    initializeVector(B, N);

    std::cout << "Vector A:" << std::endl;
    printVector(A, N);
    std::cout << "Vector B:" << std::endl;
    printVector(B, N);

    vecAdd(A, B, C, N);
    std::cout << "\nResulting Vector C:" << std::endl;
    printVector(C, N);
    return 0;
}