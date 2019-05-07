'''
this is a basic cluster algorithm : k-means.
使用西瓜数据集 4.0
'''
import pandas as pd 
import numpy as np 

'''
随机初始化k个初始均值
'''
def init_means( data, k ):
    size = data.shape[0]
    if k > size :
        print( "Cluster number is bigger than the size of the dataset")
        return data.values
    step = size // k
    index = [x for x in range(step-1,size,step)]
    means = df.iloc[index,:]
    return means 

'''
计算最接近的均值，作为data_item的预测分类
返回最后分类
'''
def min_distance_cluster( means, item ):
    temp = (means-item)**2
    distance = temp.sum(axis=1) 
    min_distance = distance.min()

    cluster = -1 #暂时不分类
    for i in range( distance.shape[0] ):
        if distance.iloc[i] == min_distance :
            cluster = i
    #print( "the cluster is ", cluster )
    return cluster


'''
根据data当前分类，计算各个均值
'''
def update_means( data , means ):
#    print( data )
    grouped = data.groupby(['class'])
    for cluster, group in grouped :
#        print( cluster , type(means) )
        c = group.mean( axis=0)
#        print( c , type(c) )
        means.iloc[cluster] = c   #emmmmmmmmmmmmmmmmmmmmmmmmmm
        #c = group.mean(axis=0)
    return 

'''
    means_info = data.groupby(['class']).mean()
'''

'''
k-means 算法基本框架
NP，通过不断优化迭代产生结果。
初始化均值
    根据距离度量更新分类
    根据分类更新均值
'''
def k_means_cluster( data , k ):
    means = init_means( data , k )
    print(means )
    data['class'] = -1
    for _ in range(10) :
        for i in range( data.shape[0] ):
            data.iloc[i,2] = min_distance_cluster(means, data.iloc[i])

        for i in range(k) :
            update_means( data, means )
    return means 
        
if __name__ == '__main__' :
    df = pd.read_csv("watermelon4.csv" )
    df = df.drop(['number'], axis=1)
    means = k_means_cluster( df , 4 )
    print( df )
    print( means )
'''
    means = init_means( df , 4 )

    c = df.iloc[0,:].values
    df['class'] = -1 
    df.iloc[0,2] = min_distance_cluster( means, df.iloc[0,:-1] )
    print("*"*10)

    update_means(df, means) 

'''

''' numpy 操作
    a = np.ones((4,4))
    print( a  )
    b = np.array( [1,2,3,4])
    print( b )
    print( a-b )
    print( (a-b)**2 )
'''
