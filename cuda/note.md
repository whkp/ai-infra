# cuda note
## 基础定义

GPU和CPU通信是异步的，CPU调用核函数后不会立刻执行，而是会返回一个事件，GPU会**异步执行**核函数；可以通过调用**cudaDeviceSynchronize()等待事件结束;**

__gloabal__用于定义核函数，GPU上执行，从CPU端通过三重尖括号语法调用，可以有参数，不可以有返回值；

__device__用于定义设备函数，GPU上执行，从GPU端调用，main函数无法直接调用，核函数调用；不需要有三重尖括号语法调用,可以有参数，可以有返回值；

__host__用于定义主机函数，CPU上执行，不需要有三重尖括号语法调用,可以有参数，可以有返回值,可以省略，默认为__host__；

可以同时定义__host__和__device_ _;

## 获取线程编号

threadIdx_x = threadIdx.x; // 线程编号x
blockIdx_x = blockIdx.x;  // 块编号x

cuda中有比线程更大的概念，叫板块（block);

每个板块有多个线程，线程编号从0开始，板块编号从0开始；

**thread 和 block 之间的关系：**
**当前线程在板块的编号：threadIdx.x; thread可以是三维的**
**当前板块中的线程数量：blockDim.x;**
**当前板块的编号：blockIdx.x;**
**总的板块数量：gridDim.x;**

![](../pic/1.png)

线程：并行的最小单位
板块：包含多个线程
网络：整个任务，包含多个板块
kernel调用语法：<<<griddim,blockdim>>>

## cuda内存

GPU使用独立显存，不能访问CPU内存；要使用显存的地址，cuda核函数才能够使用；

cuda分配内存：cudaMalloc,  cudaFree

cuda内存拷贝：cudaMemcpy,会自动进行同步

分配统一内存：cudaMallocManaged，分配的地址在CPU和GPU是一模一样的，都可以直接访问同一快内存

## C++式编程

CPU到GPU的传参一般按值传递

thrust包装了很多C++用法，会根据在CPU和GPU上执行具体的操作

支持原子操作（操作时只能一个线程操作，读取加法写回操作不会有其他线程打扰

atomicAdd(dst, src),会返回旧值

常见的原子操作：

atomicAdd(dst, src): *dst += src

atomicSub(dst, src): *dst -= src

atomicOr(dst, src): *dst |= src

atomicAnd(dst, src): *dst &= src

atomicXor(dst, src): *dst ^=src

atomicMax(dst, src): *dst = std::max( *dst, src)

atomicExch:原子写入并读取旧值；

atomicCAS: 可以实现任意的原子操作

old = atomicCAS(dst, cmp, src)相当于：

old = *dst;

if(old  == cmp) *dst = src;

由于原子操作要保证同一时刻只能有一个线程修改地址，多个线程同时修改就需要去排队，一个线程修改完后另一个线程才能进去

## 板块与共享内存

GPU由多个流式多处理器（SM）组成，每个SM可以处理一个或多个板块；

SM由多个流式单处理器（SP）组成，每个SP可以处理一个或多个线程；

SM都有自己的一个共享内存；

