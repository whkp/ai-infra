#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>

//std：：vector作为模板类，有两个参数std::vector<T, std::allocator<T>>，第一个参数是元素类型，第二个参数是分配器类型
//allocator会去调用 allocate和 deallocate 函数来分配和释放内存

template <typename T>
struct CudaAllocator {
    /* data */
    using value_type = T;
    T* allocate(size_t size) {
        T* ptr = nullptr;
        //cudaMalloc会出错，因为vector默认初始化，不在主存上会报错
        checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }

    void deallocate(T* ptr, size_t size = 0) {
        checkCudaErrors(cudaFree(ptr));
    }
};

__global__ void kernel2(int *arr, int n) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        arr[i] = i;
    }
}

template <class Func>
__global__ void parallel_for(int n, Func func) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

int main() {
    // kernel2:
    // int n =  65535;
    // //arr指向的是GPU内存，但是是在堆栈上的变量
    // std::vector<int, CudaAllocator<int>> arr(n);
    // //用arr.data()获取指向GPU内存的指针，kernel第一个参数为int*
    // kernel2<<<32, 256>>>(arr.data(), n);
    // checkCudaErrors(cudaDeviceSynchronize());
    // for(int i = 0; i < n; i++) {
    //     printf("arr[%d]: %d\n", i, arr[i]);
    // }
    // return 0;

    int n = 65535;
    float a = 3.14f;
    std::vector<float, CudaAllocator<float>> x(n);
    std::vector<float, CudaAllocator<float>> y(n);

    for(int i = 0; i < n; i++) {
        x[i] = std::rand() * (1.0f / RAND_MAX);
        y[i] = std::rand() * (1.0f / RAND_MAX);
    }
    //值传递
    parallel_for<<<32, 256>>>(n, [a, x = x.data(), y = y.data()] (int i) {
        x[i] = a * x[i] + y[i];
    });
    checkCudaErrors(cudaDeviceSynchronize());
    for(int i = 0; i < n; i++) {
        printf("x[%d]: %f\n", i, x[i]);
    }
    return 0;
}