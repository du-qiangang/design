
import math
import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
import pycuda.tools
from pycuda.compiler import SourceModule
import threading


blocksize=32
variables={}
trainable_variables={}
embeddings=[]


def get_collection(name):
    return variables[name]


model = SourceModule("""
__global__ void l2_normalize(float *x, int size, int width)
{
    int tx=threadIdx.x + blockIdx.x * blockDim.x;
    int ty=threadIdx.y + blockIdx.y * blockDim.y;
    int i=threadIdx.x;
    __shared__ float dis[32];
    int idx=width*ty+tx;
    if (tx<size){
        dis[tx]+=x[idx]*x[idx];
    }
    __syncthreads();
    if(ty==0){
        dis[tx]=rsqrtf(dis[tx]);
    }
    __syncthreads();
    x[idx]=x[idx]*dis[tx];
}
""")


def l2_normalize(x,dim,epsilno=1e-12):
    #按列求范数
    width=x.shape[1]
    height=x.shape[0]
    global embeddings
    if dim==0:
        func=model.get_function('l2_normalize')
        block=(blocksize, blocksize, 1)
        grid=((width - 1) / blocksize + 1, (height - 1) / blocksize + 1)
        func(x,height, width,
             block=block,grid=grid)
        embeddings=np.zeros_like(x)
        drv.memcpy_dtoh(embeddings,x)
        return embeddings
    else:
        if dim==1:
            func = model.get_function('l2_normalize')
            block = (blocksize, blocksize, 1)
            grid = ((width - 1) / blocksize + 1, (height - 1) / blocksize + 1)
            func(x, width, width,
                 block=block, grid=grid)
            embeddings = np.zeros_like(x)
            drv.memcpy_dtoh(embeddings, x)
            return embeddings
    if dim==3:
        return


def exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True):
    dec_rates=[]
    if staircase:
        while global_step>0:
            exp=math.ceil(global_step / decay_steps)
            decayed_learning_rate = learning_rate * decay_rate ** exp
            dec_rates.append(decayed_learning_rate)
            global_step-=1
    else:
        while global_step>0:
            exp=global_step / decay_steps
            decayed_learning_rate = learning_rate * decay_rate ** exp
            dec_rates.append(decayed_learning_rate)
            global_step-=1
    return dec_rates



def batch(data,batch_size,shapes):
    image=data[:shapes[0][0],:shapes[0][1],:]
    labels=data[shapes[0][0]:,shapes[0][1]:,:]
    return image,labels


def batch_join(data_list,batch_size,capacity=32,enqueue_many=False,shapes=None,allow_smaller_final_batch=False):
    thread_num=len(data_list)
    if enqueue_many:
        imgs=[]
        labs=[]
        for i in range(thread_num):
            img,lab=batch(data_list,batch_size,shapes)
            imgs.append(img)
            labs.append(lab)
        return imgs,labs
    else:
        imgs=data_list[:shapes[0][0],:shapes[0][1],:]
        labs=data_list[shapes[0][0]:,shapes[0][1]:,:]
        return imgs,labs


