#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mma.h"
#include "stdio.h"
#include "stdlib.h"
#include <stdio.h>
#include "math.h"
#include <vector>
#include "random"
#include "chrono"
#include <iomanip>
#include "fstream"
#include "string"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#define N (1024 * 1024)
#define BLOCK_SIZE 32
#define SYNC_D cudaDeviceSynchronize();


#define CUDA_CHECK(call) \
do { \
cudaError_t err__ = (call); \
if (err__ != cudaSuccess) { \
fprintf(stderr, "Ошибка CUDA в %s:%d — %s\n", \
__FILE__, __LINE__, cudaGetErrorString(err__)); \
exit(EXIT_FAILURE); \
} \
} while (0)



using namespace std;


float check_diff(float* v1,float* v2, int n, float tol = 0.000001f)
{
    float diff = 0;
    for (int i = 0; i < n; i++)
    {
        diff += abs(v1[i] - v2[i]);
    }
    if (diff > tol)
    {
        cout << "Error! v1 and v2 are not equal\n";
    }
    else
    {
        cout << "OK! v1 and v2 are equal\n";
    }
    return diff;

}

void check_corr(const float* h_a,   const float* h_b, float* h_c)
{
    int errors = 0;
    float maxDiff = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i]; // то же действие, что и в ядре
        float diff = fabsf ( expected - h_c[i] );
        if (diff > maxDiff) maxDiff = diff;
        if (diff > 1e-4f) errors++;
    }
    printf ( "Расхождений: %d из %d, максимальное отклонение: %e\n",
             errors, N, maxDiff );
    printf ( errors == 0 ? "Результат верен.\n"
                         : "ОШИБКА: результат не совпадает с эталоном.\n" );
}

void print_n_vec(float* ve, int n)
{
    for(int i = 0; i < n; i++)
    {
        cout << std::setprecision(4) << ve[n-i-1] << "  ";
    }
    cout << "\n";
}
template<typename T>
void rand_vec(T v, int n)
{
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution < float > dist(0, 10);
    for (int i = 0; i < n; i++)
    {
        v[i] = dist(gen);
    }
}

__global__ void kerr(float* data, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
    {
        float x = 2.0f * 3.1415926f * idx / n;
        data[idx] = sinf(sqrtf(x));

    }
}

__global__ void addKer(const float* v1, const float* v2, float* v3, int n)
{
    size_t ind = blockDim.x * blockIdx.x + threadIdx.x;
    if (ind < n)
    {
        v3[ind] = v1[ind] + v2[ind];
    }
}
void cpu_add(const float* v1, const float* v2, float* v3, int n, ofstream& f)
{

    auto start = chrono::high_resolution_clock::now();

    for(int ind = 0; ind < n; ind++)
    {
        v3[ind] = v1[ind] + v2[ind];
    }
    auto end = chrono::high_resolution_clock::now();

    cout << "time elapsed in CPU ms: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
    f << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;
}
void save_data(ofstream f, float* ve, int n)
{
    for(int i = 0; i < n; i++)
    {
        f << to_string(ve[i]) + "; ";
    }
    f << endl;
}

void task1();
void task2();
void task3();
void task4();

int main()
{
    task1();
    task2();
    task3();
    task4();


    return 0;
}

void task1()
{

    float* h_a = (float*)malloc(sizeof(float) * N);
    float* d_a;
    CUDA_CHECK(cudaMalloc((void**)&d_a, sizeof(float) * N));

    int grid_size = N / BLOCK_SIZE + (N % BLOCK_SIZE > 0);

    kerr <<< grid_size, BLOCK_SIZE >>> (d_a, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_a, d_a, N, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; i++)
    {
        printf("a[%d] = %.5f\n", i, h_a[i]);
    }


    int deviceCount = 0;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s).\n\n";

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        // Populate the property structure for the current device index
        cudaGetDeviceProperties(&prop, i);

        std::cout << "--- Device " << i << ": " << prop.name << " ---\n";
        std::cout << "  Name:          " << prop.name << "\n";

        std::cout << "  Compute Capability:          " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total Global Memory:         " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "  Shared Memory per Block:     " << prop.sharedMemPerBlock / 1024 << " KB\n";
        std::cout << "  Registers per Block:     " << prop.regsPerBlock << "\n";

        std::cout << "  Warp Size:                   " << prop.warpSize << "\n";
        std::cout << "  Max Threads per Block:       " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Threads per SM:   " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Const Memory:         " << prop.totalConstMem / 8 << " B\n";


        std::cout << "  Cuda-cores amount         " << prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor << "\n";

    }



    CUDA_CHECK(cudaFree(d_a));
    free(h_a);
}

void task2()
{
    vector<int> Ns = { (int)pow(2, 16),(int)pow(2, 18),(int)pow(2, 20),(int)pow(2, 22),(int)pow(2, 24) };
    ofstream cpu_time, gpu_total, gpu_ker;
    cpu_time.open("../cpu.csv");
    gpu_total.open("../gpu_total.csv");
    gpu_ker.open("../gpu_ker.csv");

    cpu_time << "N; time (ms)\n";
    gpu_total << "N; time (ms)\n";
    gpu_ker << "N; time (ms)\n";

    for (int i = 0; i < Ns.size(); i++)
    {
        cpu_time << to_string(Ns[i]) + "; ";
        gpu_total << to_string(Ns[i]) + "; ";
        gpu_ker << to_string(Ns[i]) + "; ";

        int n = Ns[i];
        auto vec1h = new float[n];
        auto vec2h = new float[n];
        auto vec3h = new float[n];
        auto vec3hh = new float[n];
        float *vec1d, *vec2d, *vec3d;

        rand_vec(vec1h, n);
        rand_vec(vec2h, n);
        cout << "2^N, N = " << 16 + i * 2 << endl;

        print_n_vec(vec1h, 16);
        print_n_vec(vec2h, 16);

        cudaEvent_t start1, end1;
        cudaEventCreate(&start1);
        cudaEventCreate(&end1);
        cudaEvent_t start2, end2;
        cudaEventCreate(&start2);
        cudaEventCreate(&end2);

        dim3 grid_size(n / BLOCK_SIZE + ((n % BLOCK_SIZE) > 0));

        SYNC_D
        cudaEventRecord(start1);
        cudaMalloc((void**) &vec1d, sizeof(float) * n);
        cudaMalloc((void**) &vec2d, sizeof(float) * n);
        cudaMalloc((void**) &vec3d, sizeof(float) * n);

        cudaMemcpy(vec1d, vec1h, sizeof(float) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(vec2d, vec2h, sizeof(float) * n, cudaMemcpyHostToDevice);


        cudaEventRecord(start2);

        addKer<<<grid_size, BLOCK_SIZE>>>(vec1d, vec2d, vec3d, n);

        cudaEventRecord(end2);

        cudaMemcpy(vec3h, vec3d, sizeof(float) * n, cudaMemcpyDeviceToHost);
        cudaEventRecord(end1);
        cudaEventSynchronize(end1);
        SYNC_D
        print_n_vec(vec3h, 16);
        cout << endl;

        cout << cudaGetLastError() << endl;

        float gpu_time = 0;

        cudaEventElapsedTime(&gpu_time, start1, end1);
        cout << "total time elapsed in GPU ms: " << gpu_time << endl;
        gpu_total << gpu_time << endl;

        cudaEventElapsedTime(&gpu_time, start2, end2);
        cout << "time elapsed in GPU kernel ms: " << gpu_time << endl;
        gpu_ker  << gpu_time << endl;

        cpu_add(vec1h, vec2h, vec3hh, n, cpu_time);

        check_diff(vec3hh, vec3h, n);
        print_n_vec(vec3hh, 16);
        cout << endl;

        delete[] vec1h;
        delete[] vec2h;
        delete[] vec3h;
        delete[] vec3hh;

        cudaFree(vec1d);
        cudaFree(vec2d);
        cudaFree(vec3d);
        cudaEventDestroy(start1);
        cudaEventDestroy(end1);
        cudaEventDestroy(start2);
        cudaEventDestroy(end2);
    }
    cpu_time.close();
    gpu_total.close();
    gpu_ker.close();
}

void task3()
{

    vector<int> Bs = { (int)pow(2, 5),(int)pow(2, 6),(int)pow(2, 7),(int)pow(2, 8),(int)pow(2, 9), (int)pow(2, 10), 100, 200, 555};
    ofstream gpu_ker;


    gpu_ker.open("../gpu_kerB.csv");

    gpu_ker << "N; time (ms)\n";

    for (int i = 0; i < Bs.size(); i++)
    {
        int n = (int)pow(2, 24);
        auto vec1h = new float[n];
        auto vec2h = new float[n];
        auto vec3h = new float[n];
        auto vec3hh = new float[n];
        float *vec1d, *vec2d, *vec3d;

        rand_vec(vec1h, n);
        rand_vec(vec2h, n);
        cout << "BLOCK_SIZE " << Bs[i] << endl;
        gpu_ker << Bs[i] << "; ";

        print_n_vec(vec1h, 16);
        print_n_vec(vec2h, 16);


        cudaEvent_t start2, end2;
        cudaEventCreate(&start2);
        cudaEventCreate(&end2);

        dim3 grid_size(n / Bs[i] + ((n % Bs[i]) > 0));


        cudaMalloc((void**) &vec1d, sizeof(float) * n);
        cudaMalloc((void**) &vec2d, sizeof(float) * n);
        cudaMalloc((void**) &vec3d, sizeof(float) * n);

        cudaMemcpy(vec1d, vec1h, sizeof(float) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(vec2d, vec2h, sizeof(float) * n, cudaMemcpyHostToDevice);


        cudaEventRecord(start2);
        SYNC_D

        addKer<<<grid_size, Bs[i]>>>(vec1d, vec2d, vec3d, n);
        SYNC_D

        cudaEventRecord(end2);

        cudaMemcpy(vec3h, vec3d, sizeof(float) * n, cudaMemcpyDeviceToHost);

        print_n_vec(vec3h, 16);
        cout << endl;

        cout << cudaGetLastError() << endl;


        float gpu_time = 0;


        cudaEventElapsedTime(&gpu_time, start2, end2);
        cout << "time elapsed in GPU kernel ms: " << gpu_time << endl;
        gpu_ker << gpu_time << endl;
        cout << endl;

        delete[] vec1h;
        delete[] vec2h;
        delete[] vec3h;
        delete[] vec3hh;

        cudaFree(vec1d);
        cudaFree(vec2d);
        cudaFree(vec3d);
        cudaEventDestroy(start2);
        cudaEventDestroy(end2);

    }
    gpu_ker.close();
}

void task4()
{
    vector<int> Ns = { (int)pow(2, 16),(int)pow(2, 18),(int)pow(2, 20),(int)pow(2, 22),(int)pow(2, 24) };
    ofstream gpu_ker;

    gpu_ker.open("../gpu_kerC.csv");
    gpu_ker << "N; ker_time; total_time\n";
    for (int i = 0; i < Ns.size(); i++)
    {
        cudaEvent_t  cstart, cend;
        cudaEventCreate(&cstart);
        cudaEventCreate(&cend);
        float ker_time = 0;

        int n = Ns[i];
        thrust::host_vector<float> ha(n), hb(n), hc(n);
        rand_vec(ha, n);
        rand_vec(hb, n);

        gpu_ker << to_string(Ns[i]) + "; ";

        auto start = chrono::high_resolution_clock::now();
        SYNC_D
        thrust::device_vector<float> da = ha, db = hb, dc(n);
        cudaEventRecord(cstart);
        thrust::transform(da.begin(), da.end(), db.begin(), dc.begin(), thrust::plus<float>());
        cudaEventRecord(cend);
        hc = dc;
        SYNC_D
        auto end = chrono::high_resolution_clock::now();
        cudaEventElapsedTime(&ker_time, cstart, cend);
        gpu_ker << ker_time << "; ";
        auto time_elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        gpu_ker << ((float)time_elapsed) << endl;



    }

    gpu_ker.close();
}