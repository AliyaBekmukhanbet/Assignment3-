%%cuda
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000   // Размер массивов

// CUDA-ядро: каждый поток складывает один элемент массивов
__global__ void add_arrays(float *d_a, float *d_b, float *d_c, int n) {

    // Глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка границ для последнего неполного блока
    if (idx < n) {
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}

// Функция для измерения времени выполнения ядра
// при заданном размере блока
void test_performance(int block_size,
                      float *d_a, float *d_b, float *d_c, int n) {

    // CUDA Events используются для точного тайминга GPU-ядра
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Количество блоков рассчитывается так,
    // чтобы покрыть все элементы массива
    int grid_size = (n + block_size - 1) / block_size;

    // Запуск CUDA-ядра с заданным размером блока
    add_arrays<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

    // Ожидание завершения ядра и измерение времени
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Block size: %d, Время выполнения: %f мс\n",
           block_size, milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {

    // Память на хосте (CPU) и устройстве (GPU)
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Инициализация входных данных на CPU
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Копирование данных с CPU на GPU
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Тестирование разных размеров блока
    // для анализа влияния blockDim на производительность
    int block_sizes[] = {128, 256, 512};
    int num_tests = sizeof(block_sizes) / sizeof(int);

    for (int i = 0; i < num_tests; i++) {

        // Очистка выходного массива (необязательно, для чистоты эксперимента)
        cudaMemset(d_c, 0, N * sizeof(float));

        test_performance(block_sizes[i], d_a, d_b, d_c, N);
    }

    // Копирование результата последнего запуска (опционально)
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Освобождение памяти
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
