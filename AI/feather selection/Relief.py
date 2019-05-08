'''
进行特征选择
这里根据Relief模型，计算出各个特征的相关统计量
基于西瓜数据集3.0
'''


import pandas as pd 
import numpy as np 

'''
连续数据进行规范化
'''
def preprocess( data ):
    data = data.drop(['编号'], axis=1)
    continue_list = ['density','ratio_suger']
    for attr in continue_list :
        min_ = data[attr].min()
        max_ = data[attr].max()
        data[attr] = (data[attr] - min_) / (max_ - min_)
    return data

'''
near-hit
near-miss
find the near item in data with respect to goal
return the near item.
'''
def near_item( df , goal ):
    distinct_attr = ['色泽','根蒂','敲声','纹理','脐部','触感']
    data = df.drop('label', axis=1)
    
    for attr in distinct_attr :
        for i in range( data.shape[0]):
            if (data[attr].iloc[i] == goal[attr]) :
                data[attr].iloc[i] = 0
            else :
                data[attr].iloc[i] = 1

    data['density'] -= goal['density']
    data['ratio_suger'] -= goal['ratio_suger']

    distance = data.sum(axis=1)
    #print( distance )
    for i in range( distance.shape[0] ):
        if distance.iloc[i] == distance.min() :
            #print( i )
            #print( distance.iloc[i])
            return data.iloc[i] #处理后的各项差

'''
get the dataframe for near hit difference.
'''
def hit( data ):
    near_hit = []
    for i in range( data.shape[0] ):
        temp = data.drop( i )
        grouped = temp.groupby(['label'])
        group_0 = grouped.get_group(0)
        group_1 = grouped.get_group(1)

        if ( data.iloc[i,-1] == 1 ):
            near = near_item( group_1, data.iloc[i] )
            #print( near , type(near) )
            near_hit.append( near )
        else :
            near = near_item( group_0, data.iloc[i])
            near_hit.append( near )
    
    near_hit = np.array( near_hit )
    print( near_hit , type(near_hit)) 
    return near_hit

def miss( data ):
    near_miss = []
    for i in range( data.shape[0] ):
        temp = data.drop( i )
        grouped = temp.groupby(['label'])
        group_0 = grouped.get_group(0)
        group_1 = grouped.get_group(1)

        if ( data.iloc[i,-1] == 1 ):
            near = near_item( group_0, data.iloc[i] )
            #print( near , type(near) )
            near_miss.append( near )
        else :
            near = near_item( group_1, data.iloc[i])
            near_miss.append( near )
    
    near_miss = np.array( near_miss )
    print( near_miss , type(near_miss)) 
    return near_miss

'''
计算相关统计量

'''
def calculate_relief( data ):
    temp = data.values 
    near_hit = hit( data )
    near_miss = miss( data )
    ans = near_miss*near_miss - near_hit*near_hit
    ans = ans.sum( axis=0 )
    print( ans )
    return ans


if __name__=="__main__" :
    df = pd.read_csv("watermelon3.csv")
    df = preprocess( df )

    #hit( df )
    attr_factor = calculate_relief( df )
    print( "Relief 分析结果为： ")
    print( attr_factor )

