#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <curand.h>
#include <curand_kernel.h>

#define ull unsigned long long
#define ld long double

#define GTX_1060_BLOCKS 1280
#define WARP_SIZE 32 // количество потоков в блоке

/**
 * Запуск по всем блокам. После выполнения функции в per_blocks_sum лежат
 * сумма для каждого блока, и тогда эти значения суммируются в одно.
 * @param per_blocks_sum массив с локальной суммой по всем потокам каждого блока
 * @param iterations количество итераций на поток.
 */
__global__ void kernel(ull *per_blocks_sum, ull iterations) {
    __shared__ ull per_block_sum[WARP_SIZE];
    ull index = threadIdx.x + blockIdx.x * blockDim.x;

    curandState_t rng;
    curand_init(clock64(), index, 0, &rng);

    per_block_sum[threadIdx.x] = 0;

    for (int i = 0; i < iterations; i++) {
        double x = curand_uniform(&rng); // x в [0,1]
        double y = curand_uniform(&rng); // y в [0,1]
        per_block_sum[threadIdx.x] += 1 - int(x * x + y * y);
    }

    if (threadIdx.x == 0) {
        per_blocks_sum[blockIdx.x] = 0;
        for (int i = 0; i < WARP_SIZE; i++) {
            per_blocks_sum[blockIdx.x] += per_block_sum[i];
        }
    }
}

__host__ ld monteCarloCPU(ull N) {
    double x,y;
    ld sum = 0;
    for(int i = 0; i < N; i++){
        x = (double) rand()/RAND_MAX;
        y = (double) rand()/RAND_MAX;
        if(x*x + y*y <= 1) sum += 1.0;
    }
    return sum * 4.0 / (ld)(N);
}

__host__ ld monteCarloGPU(ull N) {
    ull iterations;
    size_t size = N * sizeof(ull);

    ull *sums_per_blocks = nullptr;

    cudaMalloc(&sums_per_blocks, size);

    iterations = N / (GTX_1060_BLOCKS * WARP_SIZE);
    if (iterations == 0) {
        iterations = 1;
        kernel<<<N, 1>>>(sums_per_blocks, iterations);
    }
    else {
        kernel<<<GTX_1060_BLOCKS, WARP_SIZE>>>(sums_per_blocks, iterations);
    }
    cudaDeviceSynchronize();

    ull *host_sums_per_blocks = (ull *) malloc(size);
    cudaMemcpy(host_sums_per_blocks, sums_per_blocks, size, cudaMemcpyDeviceToHost);

    double sum = 0;
    double sum_iterations  = GTX_1060_BLOCKS;
    if(iterations == 1) {
        sum_iterations = N;
    }

    for (int i = 0; i < sum_iterations; i++) {
        sum += host_sums_per_blocks[i];
    }
    double divizor = iterations == 1 ? N : GTX_1060_BLOCKS * WARP_SIZE * iterations;

    free(host_sums_per_blocks);
    cudaFree(sums_per_blocks);

    return sum * 4 / divizor;
}

int main() {
    unsigned long long n = 1e8;
//    scanf("%llu", &n);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    ld pi = monteCarloGPU(n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    printf("GPU Pi:: %Lf\n", pi);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time consumed for monteCarloGPU :: %3.1f ms \n", milliseconds);

    cudaEventRecord(start, 0);
    pi = monteCarloCPU(n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    printf("CPU Pi:: %Lf\n", pi);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time consumed for monteCarloCPU :: %3.1f ms \n", milliseconds);
    return 0;
}