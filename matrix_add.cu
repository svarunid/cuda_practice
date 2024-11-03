#include <cuda.h>

#include <cstdlib>
#include <ctime>
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

__global__ void matAddKernel(float *A, float *B, float *C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

void matAdd(float *A, float *B, float *C, int M, int N) {
    float *A_d, *B_d, *C_d;

    cudaCheckError(cudaMalloc((void **)&A_d, M * N * sizeof(float)));
    cudaCheckError(cudaMalloc((void **)&B_d, M * N * sizeof(float)));
    cudaCheckError(cudaMalloc((void **)&C_d, M * N * sizeof(float)));

    cudaCheckError(cudaMemcpy(A_d, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(B_d, B, M * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(cdiv(N, threadsPerBlock.x), cdiv(M, threadsPerBlock.y));
    matAddKernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, M, N);

    cudaCheckError(cudaPeekAtLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void printMatrix(const float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    int M = 100;
    int N = 49;

    float A[M * N];
    float B[M * N];
    float C[M * N];

    initializeMatrix(A, M, N);
    initializeMatrix(B, M, N);

    std::cout << "Matrix A (M x N):" << std::endl;
    printMatrix(A, M, N);
    std::cout << "\nMatrix B (M x N):" << std::endl;
    printMatrix(B, M, N);

    matAdd(A, B, C, M, N);
    std::cout << "\nResulting Matrix C (M x N):" << std::endl;
    printMatrix(C, M, N);
    return 0;
}