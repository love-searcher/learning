'''
使用k_means.py文件
不确切指定k的值，通过一定的评价标准来找一个比较合适的k值
'''
import numpy as np 
import pandas as pd 
from k_means import *


def calculate_inter_distance( df , means ):
    distance = 0
    grouped = df.groupby(['class'])
    for cluster, group in grouped :
        mean = means.iloc[cluster,:]
        diff = (group-mean)**2
        diff = diff.sum(axis=1)
        distance += diff.sum()
    return distance 

'''
不事先指定k，自动选择一个合适的k值
这个需要对每次k进行一个评价
评价标准用： 类内误差。// 虽然不太合理 emm, 任意多个的时候emm
input : dataframe of dataset
output : k and the means
'''
def d_k_means( df ):
    k = 1
    max_total_inter_distance = float('inf')

    while True :
        means = k_means_cluster( df , k )
        total_inter_distance = calculate_inter_distance( df, means )
        if ( k > 1 ):
            print("*"*100 )
        print( k , "  " , total_inter_distance )
        if total_inter_distance < max_total_inter_distance :
            max_total_inter_distance = total_inter_distance
        else :
            break
        k = k+1
    means = k_means_cluster( df , k )
    return k,means 

if __name__ == "__main__" :
    df = pd.read_csv("watermelon4.csv" )
    df = df.drop(['number'], axis=1)
    k, means = d_k_means( df )
    print( k , means )

    #means = k_means_cluster( df , 4)
    #distance = calculate_inter_distance( df , means )
    #print( 4 ,  distance  )

