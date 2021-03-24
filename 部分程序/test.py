from __future__ import print_function
from __future__ import absolute_import
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import os


def main():
    data=np.loadtxt('D:/毕业论文/log.txt')
    train_time=data[:,3].tolist()
    triplets_num=data[:,0].tolist()
    select_time=data[:,1].tolist()
    k=0.0070118904
    train_t=((k*128)/(160*160))
    r=random.uniform(k-0.0009,k+0.0009)
    rate = random.uniform(0.6310, 0.6340)
    x=np.arange(1, data.shape[0])
    l=plt.plot(x,train_time,'b--',label='train_time')
    plt.title('train_facenet')
    plt.xlabel('epoch')
    plt.ylabel('time/s')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()







