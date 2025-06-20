#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__device__ void say_hello() {
    printf("Hello from CUDA device!\n");
}

__global__ void kernel() {
    printf("thread %d of %d\n", threadIdx.x, blockDim.x);
    //获取总的线程编号
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tnum = blockDim.x * gridDim.x;
    printf("flattened thread %d of %d\n", tid, tnum);
}

__global__ void kernel2() {
    say_hello();
}

__global__ void kernel3(int *pret) {
    *pret = 42;
}

//网格跨步循环
__global__ void kernel4(int *arr, int n) {
    //计算当前线程在全局线程索引中的位置
    //每次加的数量是blockDim.x * gridDim.x，确保每个线程都能处理不同的索引
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        //将当前线程索引赋值给数组
        arr[i] = i;
    }
}

int main() {
    //kernel<<<2, 3>>>();
    //可以是3个参数
    // kernel<<dim3(2,3,1), dim3(1,1,1)>>>();
    int n = 65535;
    int *arr;
    checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
    kernel4<<<32, 1024>>>(arr, n);
    checkCudaErrors(cudaDeviceSynchronize());
    for(int i = 0; i < n; i++) {
        printf("%d\n ", arr[i]);
    }
    cudaFree(arr);
    return 0;
}