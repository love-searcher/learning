import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import math

continue_attributes = ["density" , "ratio_suger"]

'''
statics the prob for distinct attribute of value in data
'''
def distinct_statics(data, attribute, value):
    size,_ = data.shape
    #grouped = data.groupby([attribute])
    c = data[data[attribute]==value]
    new_size,_ = c.shape
    return new_size/size

def continue_statics(data, attribute, value):
    mean = data[attribute].mean()
    std = data[attribute].std()
    p = 1/(math.sqrt(2*math.pi)*std) * (math.exp((-1*(value-mean)**2)/(2*std**2)))
    return p

'''
using naive bayes to predict the class
input the sample and the dataset
output the class
'''
def naive_bayes( data, sample ):
    max_prob = -1
    max_class = ""
    total_size,_ = data.shape
    grouped = data.groupby(["label"])
    for val, group in grouped :
        # the possibility of the class
        group_size,_ = group.shape
        p_class = group_size / total_size
        
        p = 1
        for attribute in sample.columns:
            value = sample[attribute][0]
            if attribute in continue_attributes :
                p *= continue_statics(group, attribute, value)
            else:
                p *= distinct_statics(group, attribute, value)
        p *= p_class 
        if p > max_prob :
            max_prob = p 
            max_class = val
        print( val , ' : ' , max_class , p)
    return max_class

if __name__ == '__main__' :
    df = pd.read_csv("watermelon3.csv")
    data = df.drop(['编号'],axis=1)
    sample = pd.read_csv("sample.csv")
    print( sample )
    
    ans = naive_bayes( data , sample )
    print( ans )
