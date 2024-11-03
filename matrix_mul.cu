#include <cuda.h>

#include <cstdlib>
#include <ctime>
#include <iostream>

constexpr int TILE_SIZE = 16;

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

__global__ void matMulKernel(float *A, float *B, float *C, size_t pitch_A, size_t pitch_B,
                             size_t pitch_C, size_t M, size_t K, size_t N) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    float value = 0.0f;
    for (int blk = 0; blk < ((K + TILE_SIZE - 1) / TILE_SIZE); ++blk) {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        int A_row = blockRow * TILE_SIZE + row;
        int A_col = blk * TILE_SIZE + col;
        As[row][col] =
            A_row < M && col < K ? *((float *)((char *)A + A_row * pitch_A) + A_col) : 0.0f;

        int B_row = blk * TILE_SIZE + col;
        int B_col = blockCol * TILE_SIZE + row;
        Bs[row][col] =
            B_col < N && row < K ? *((float *)((char *)B + B_row * pitch_B) + B_col) : 0.0f;
        __syncthreads();

        for (int e = 0; e < TILE_SIZE; ++e) {
            value += As[row][e] * Bs[e][col];
        }
        __syncthreads();
    }

    int C_row = blockRow * TILE_SIZE + row;
    int C_col = blockCol * TILE_SIZE + col;
    if (C_row < M && C_col < N) {
        *((float *)((char *)C + C_row * pitch_C) + C_col) = value;
    }
}

void matMul(float *A, float *B, float *C, size_t M, size_t K, size_t N) {
    float *A_d, *B_d, *C_d;
    size_t pitch_A, pitch_B, pitch_C;

    cudaCheckError(cudaMallocPitch((void **)&A_d, &pitch_A, K * sizeof(float), M));
    cudaCheckError(cudaMallocPitch((void **)&B_d, &pitch_B, N * sizeof(float), K));
    cudaCheckError(cudaMallocPitch((void **)&C_d, &pitch_C, N * sizeof(float), M));

    cudaCheckError(cudaMemcpy2D(A_d, pitch_A, A, K * sizeof(float), K * sizeof(float), M,
                                cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy2D(B_d, pitch_B, B, N * sizeof(float), N * sizeof(float), K,
                                cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(cdiv(N, threadsPerBlock.x), cdiv(M, threadsPerBlock.y));
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, pitch_A, pitch_B, pitch_C, M, K,
                                                     N);

    cudaCheckError(cudaPeekAtLastError());

    cudaCheckError(cudaMemcpy2D(C, N * sizeof(float), C_d, pitch_C, N * sizeof(float), M,
                                cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(A_d));
    cudaCheckError(cudaFree(B_d));
    cudaCheckError(cudaFree(C_d));
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

    int M = 120;
    int K = 150;
    int N = 100;

    float A[M * K];
    float B[K * N];
    float C[M * N];

    initializeMatrix(A, M, K);
    initializeMatrix(B, K, N);

    std::cout << "Matrix A (M x K):" << std::endl;
    printMatrix(A, M, K);
    std::cout << "\nMatrix B (K x N):" << std::endl;
    printMatrix(B, K, N);

    matMul(A, B, C, M, K, N);

    std::cout << "\nResulting Matrix C (M x N):" << std::endl;
    printMatrix(C, M, N);
    return 0;
}