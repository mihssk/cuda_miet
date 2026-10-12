#include <iostream>
#include "cuda_runtime.h"

#define SYNC_D cudaDeviceSynchronize();
#define BLOCK_SIZE 32

void __global__ vec_sum(float* a, float* b, float* c, int N)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < N)
    {
        c[ind] = a[ind] + b[ind];
    }

}

void print_vec(float* v, int N)
{

    for(int i = 0; i < N; i++)
    {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    SYNC_D
    int N = 100;
    float* ha = new float[N];
    float* hb = new float[N];
    float* hc = new float[N];

    for(int i = 0; i < N; i++)
    {
        ha[i] = i;
        hb[i] = i;
    }

    float* da;
    float* db;
    float* dc;

    cudaMalloc((void**) &da, sizeof(float) * N);
    cudaMalloc((void**) &db, sizeof(float) * N);
    cudaMalloc((void**) &dc, sizeof(float) * N);

    cudaMemcpy(da, ha, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(float) * N, cudaMemcpyHostToDevice);

    print_vec(ha, N);
    print_vec(hb, N);

    dim3 grid_size((N / BLOCK_SIZE + ((N % BLOCK_SIZE) != 0)));
    vec_sum<<<BLOCK_SIZE, grid_size>>>(da, db, dc, N);

    cudaMemcpy(hc, dc, sizeof(float) * N, cudaMemcpyDeviceToHost);
    print_vec(hc, N);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    delete[] ha;
    delete[] hb;
    delete[] hc;
    return 0;

}
