import numpy as np 
import pandas as pd 
import math

continue_attributes = ["density" , "ratio_suger"]

'''
statics the prob for distinct attribute of value in data
'''
def distinct_statics(data, attribute, value):
    size,_ = data.shape
    grouped = data.groupby([attribute])
    count = 0 # for the different types of the attribute
    new_size = 0 #may be no instance in data for the attribute value
    for val, group in grouped :
        count += 1
        if val == value :
            new_size,_ = group.shape
    return (new_size+1)/(size+count)


def continue_statics(data, attribute, value):
    mean = data[attribute].mean()
    std = data[attribute].std()
    p = 1/(math.sqrt(2*math.pi)*std) * (math.exp((-1*(value-mean)**2)/(2*std**2)))
    return p

'''
using AODE to predict the class
input the sample and the dataset
output the class
'''
def AODE( data, sample ):
    p1 = 0
    p2 = 0
    max_class = ""
    total_size,_ = data.shape
    for attribute in data.columns[:-1] :
        if attribute not in continue_attributes :
            grouped = data.groupby(["label",attribute])
            count = 0 
            total_prob = 0
            for _ in grouped :
                count += 1
            # the attribute as the sup-fathor
            for val, group in grouped :
                temp = group[attribute].values[0]
                temp1 = sample[attribute].values[0]
                if ( temp != temp1):# the x_i satisfy the sample
                    continue
                size,_ = group.shape
                p = 1
                for item in data.columns[:-1]: # calculate the conditional prob
                    if item == attribute : # the sup-father
                        continue
                    if item in continue_attributes :
                        p *= continue_statics(group, item, sample[item][0]) # item is an attribute
                    else:
                        p *= distinct_statics(group, item, sample[item][0])
                p *= (size+1)/(total_size+count) #*P(c_i,x_i)

                t = group['label']
                print( type(t ) )
                if group['label'].values[0] == 1 :
                    p1 += p
                else :
                    p2 += p
                print( p1 , p2)# 和不是1，单独一项大于1， 原因是：没有进行归一化
    if p1 > p2 :
        return 1 #hao gua
    else :
        return 0 # huai gua

if __name__ == '__main__' :
    df = pd.read_csv("watermelon3.csv")
    data = df.drop(['编号'],axis=1)
    sample = pd.read_csv("sample.csv")
    print( sample )
    
    ans = AODE( data , sample )
    print( ans )
