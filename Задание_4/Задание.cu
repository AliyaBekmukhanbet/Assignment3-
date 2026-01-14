%%cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000   // Размер обрабатываемых массивов

// CUDA-ядро: каждый поток складывает один элемент массивов
__global__ void add_arrays(float *d_a, float *d_b, float *d_c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка границ для последнего неполного блока
    if (idx < n) {
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}

// Функция измерения времени выполнения ядра
// для заданного размера блока
float test_performance(int block_size,
                       float *d_a, float *d_b, float *d_c, int n) {

    // CUDA Events — стандартный способ точного тайминга GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Конфигурация сетки зависит от размера блока
    int grid_size = (n + block_size - 1) / block_size;

    // Запуск ядра с выбранным block_size
    add_arrays<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {

    // Память на CPU и GPU
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Инициализация входных данных
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Копирование данных на GPU
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Набор размеров блока для эксперимента
    // (от одного варпа до максимального размера блока)
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    // Поиск оптимального block_size по времени выполнения
    float min_time = 1e6;
    int optimal_block_size = block_sizes[0];

    for (int i = 0; i < sizeof(block_sizes)/sizeof(int); i++) {

        // Очистка выходного массива для чистоты эксперимента
        cudaMemset(d_c, 0, N * sizeof(float));

        float time = test_performance(block_sizes[i],
                                      d_a, d_b, d_c, N);

        printf("Block size: %d, Время выполнения: %f мс\n",
               block_sizes[i], time);

        // Выбор конфигурации с минимальным временем
        if (time < min_time) {
            min_time = time;
            optimal_block_size = block_sizes[i];
        }
    }

    // Итог эксперимента
    printf("Оптимальный размер блока: %d, время: %f мс\n",
           optimal_block_size, min_time);

    // Освобождение ресурсов
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
