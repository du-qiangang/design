import pycuda.driver as drv
import pycuda
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
import pycuda.tools
import cv2
import random

blocksize=32

class Image:
    def __init__(self,_data):
        self.data=_data
        self.shape=self.data.shape

    def value(self):
        return self.data

    def rehsape(self,_shape):
        self.data=self.data.reshape(_shape)


model = SourceModule("""
__global__ void reduce_sum(float *a, float *out, int N)
{
    int col=threadIdx.x + blockIdx.x * blockDim.x;
    int row=blockIdx.x;
    int index=col*row;
    float sum=0;
    if (col >= N)
    {
        return;
    }
    for (int i = 0;i < N; i++){
        sum += a[index+i];   
    }
    out[row]=sum;
}
__global__ void subtract(float *a, float *b, float *out, int N)
{
    int tx=threadIdx.x + blockIdx.x * blockDim.x;
    int ty=threadIdx.y + blockIdx.y * blockDim.y;
    int index=tx*(ty+1);
    if(index<N){
        out[index]=a[index]-b[index];
    }
}
__global__ void square(float *a, float *out, int N)
{
    int tx=threadIdx.x + blockIdx.x * blockDim.x;
    int ty=threadIdx.y + blockIdx.y * blockDim.y;
    int index=tx*(ty+1);
    if(index<N){
        out[index]=a[index]*a[index];
    }
}
__global__ void add(float *a, float b, float *out)
{
    int tx=threadIdx.x + blockIdx.x * blockDim.x;
    int ty=threadIdx.y + blockIdx.y * blockDim.y;
    int index=tx*(ty+1);
    out[index]=a[index]+b;
}
__global__ void maximum(float *a, float b, float *out, int N)
{
    int tx=threadIdx.x + blockIdx.x * blockDim.x;
    int ty=threadIdx.y + blockIdx.y * blockDim.y;
    int index=tx*(ty+1);
    if(index<N){
        out[index]=a[index]>b?a[index]:b; 
    }
}
__global__ void reduce_mean(const float* a, float *out, int N)
{
    int col=threadIdx.x + blockIdx.x * blockDim.x;
    int row=blockIdx.x;
    int index=col*row;
    if (col >= N)
    {
        return;
    }
    float sum=0;
    
    for (int i = 0;i < N; i++){
        sum += a[index+i];   
    }
    out[row]=sum/N;
}
""")


def reduce_sum(input_data, axis=0, keep_dims=False):
    if axis == 0:
        size = len(input_data.shape[1])
    else:
        if axis == 1:
            size = len(input_data[0])
        else:
            print("wrong parameters axis")
            return input_data
    sum_cpu = np.zeros((size,), dtype=input_data.dtype.itemsize)
    fun = model.get_function("reduce_sum")
    data_gpu = drv.mem_alloc(input_data.nbytes)
    sum_gpu = drv.mem_alloc(input_data.nbytes)
    drv.memcpy_htod(data_gpu, input_data)
    drv.memcpy_htod(sum_gpu, sum_cpu)
    threads_x = 32
    blocks_x = int((input_data.shape[0] - 1) / blocksize) + 1
    fun(data_gpu, sum_gpu, size, block=(threads_x, 1, 1), grid=(blocks_x, 1))
    drv.memcpy_dtoh(sum_cpu, sum_gpu)
    data_gpu.free()
    sum_gpu.free()
    return sum_cpu


def subtract(x, y):
    fun = model.get_function('subtract')
    sub = x.copy()
    x_gpu = drv.mem_alloc(x.nbytes)
    y_gpu = drv.mem_alloc(y.nbytes)
    sub_gpu = drv.mem_alloc(sub.nbytes)
    drv.memcpy_htod(x_gpu, x)
    drv.memcpy_htod(y_gpu, y)
    drv.memcpy_htod(sub_gpu, sub)
    threads_x = 32
    threads_y = 32
    blocks_x = (x.shape[0]-1)/blocksize+1
    blocks_y = (x.shape[1]-1)/blocksize+1
    fun(x_gpu, y_gpu, sub_gpu,x.shape[0]*x.shape[1], block=(threads_x, threads_y, 1), grid=(blocks_x, blocks_y))
    drv.memcpy_dtoh(sub, sub_gpu)
    x_gpu.free()
    y_gpu.free()
    sub_gpu.free()
    return sub


def square(x):
    fun = model.get_function("square")
    x_square = x.copy()
    x_gpu = drv.mem_alloc(x.nbytes)
    square_gpu = drv.mem_alloc(x_square.nbytes)
    drv.memcpy_htod(x_gpu, x)
    drv.memcpy_htod(square_gpu, x_square)
    threads_x = 32
    threads_y = 32
    blocks_x = (x.shape[0]-1)/blocksize+1
    blocks_y = (x.shape[1]-1)/blocksize+1
    fun(x_gpu, square_gpu, block=(threads_x, threads_y, 1), grid=(blocks_x, blocks_y))
    drv.memcpy_dtoh(x_square, square_gpu)
    x_gpu.free()
    square_gpu.free()
    return x_square


def add(x, y):
    fun = model.get_function("add")
    sum_cpu = x.copy()
    x_gpu = drv.mem_alloc(x.nbytes)
    y_gpu = drv.mem_alloc(y.nbytes)
    sum_gpu = drv.mem_alloc(sum_cpu.nbytes)
    drv.memcpy_htod(x_gpu, x)
    drv.memcpy_htod(y_gpu, y)
    drv.memcpy_htod(sum_gpu, sum_cpu)
    threads_x = 32
    threads_y = 32
    blocks_x = (x.shape[0]-1)/blocksize+1
    blocks_y = (x.shape[1]-1)/blocksize+1
    fun(x_gpu, y_gpu, sum_gpu, block=(threads_x, threads_y, 1), grid=(blocks_x, blocks_y))
    drv.memcpy_dtoh(sum_cpu, sum_gpu)
    x_gpu.free()
    y_gpu.free()
    sum_gpu.free()
    return sum_cpu


def maximum(a, b):
    if a > b:
        return a
    else:
        return b


def unstack(a, axis=0):
    list=[]
    for i in range(a.shape[0]):
        list.append(a[:,i,:])
    return list


def reduce_mean(a,  axis=None):
    if axis == 0:
        size = len(a)
        result = np.zeros((size,), dtype=np.float32)
    else:
        if axis == 1:
            size = len(a[0])
            result = np.zeros((size,), dtype=np.float32)
        else:
            size = 1
            result = a[0]
    fun = model.get_function("reduce_mean")
    result_gpu = drv.mem_alloc(result.nbytes)
    drv.memcpy_htod(result_gpu, result)
    threads = 128
    if size % threads is 0:
        blocks=threads>>7
    else:
        blocks=threads>>7+1
    fun(a, result_gpu, size, block=(threads, 1, 1), grid=(blocks, 1, 1))
    drv.memcpy_dtoh(result, result_gpu)
    return result


def exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False):
    #指数法更新学习率
    lr=[]
    return lr
