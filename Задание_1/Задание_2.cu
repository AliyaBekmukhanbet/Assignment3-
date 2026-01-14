%%cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000
#define BLOCK_SIZE 256

// CUDA-ядро: каждый поток обрабатывает один элемент массива
__global__ void multiply_shared(float *d_array, float scalar, int n) {

    // Shared memory — быстрая память, общая для потоков одного блока
    __shared__ float sdata[BLOCK_SIZE];

    // Глобальный индекс элемента массива
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Индекс потока внутри блока (для доступа к shared memory)
    int tid = threadIdx.x;

    // Загрузка данных из глобальной памяти в shared memory
    // Проверка нужна для последнего неполного блока
    if (idx < n) {
        sdata[tid] = d_array[idx];
    }

    // Синхронизация потоков блока перед использованием shared memory
    __syncthreads();

    // Вычисление в shared memory (быстрее, чем в global memory)
    if (idx < n) {
        sdata[tid] *= scalar;
    }

    // Гарантия, что все потоки завершили вычисления
    __syncthreads();

    // Запись результата обратно в глобальную память
    if (idx < n) {
        d_array[idx] = sdata[tid];
    }
}

int main() {

    // Память на CPU и GPU
    float *h_array = (float*)malloc(N * sizeof(float));
    float *d_array;
    cudaMalloc(&d_array, N * sizeof(float));

    // Инициализация на CPU и копирование на GPU
    for (int i = 0; i < N; i++) {
        h_array[i] = (float)i;
    }
    cudaMemcpy(d_array, h_array, N * sizeof(float),
               cudaMemcpyHostToDevice);

    // CUDA Events используются для точного измерения времени GPU-ядра
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Конфигурация сетки: число блоков выбирается так,
    // чтобы покрыть все элементы массива
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Запуск ядра на GPU
    multiply_shared<<<grid_size, BLOCK_SIZE>>>(d_array, 2.0f, N);

    // Завершение тайминга
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Копирование результата обратно (не влияет на измерение времени ядра)
    cudaMemcpy(h_array, d_array, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    printf("Время выполнения (shared memory): %f мс\n", milliseconds);

    // Освобождение ресурсов
    free(h_array);
    cudaFree(d_array);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
