import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def preprocess( data ):
	data = data[["density","ratio_suger","label"]]
	mean = data.groupby( "label" ).mean().values
	data_mat = data[["density","ratio_suger"]].values
	label_mat = data[["label"]].values
	return data_mat, label_mat, mean
	
#############################################
# linear discriminant for 2 class           #
# return value are the weights for the line #
#############################################
def LDA( data_mat, label_mat, mean ):	
	_,n = data_mat.shape

	S_w = np.zeros((n,n))
	for x,label in zip(data_mat, label_mat):
		temp = x-mean[label]
		S_w = S_w+(temp.T @ temp)
	
	S_w = S_w+0.01*np.ones((n,n))
	weights = np.linalg.inv(S_w)@(mean[0]-mean[1]).T
	return weights

def display( data_mat, label_mat, weights):
	m,_ = label_mat.shape
	x_cord1 = []
	y_cord1 = []
	x_cord2 = [] # to show different type of label
	y_cord2 = []
	for i in range( m ):
		if label_mat[i] == 1:
			x_cord1.append( data_mat[i,0])
			y_cord1.append( data_mat[i,1])
		else :
			x_cord2.append( data_mat[i,0])
			y_cord2.append( data_mat[i,1])
	plt.plot( x_cord1,y_cord1,'bo',x_cord2,y_cord2,'ro')
#	plt.axis( [0,1,0,1])
	x = np.arange(0,1.2,0.1)
	y = (weights[0]*x)/weights[1]
	plt.plot( x , y )
	plt.xlabel( 'density' )
	plt.ylabel( 'ratio_suger' )
	plt.show()
	
if __name__ =='__main__':
	# read data
	df = pd.read_csv("watermelon3.csv")
	data_mat, label_mat, mean = preprocess( df )

	weights = LDA( data_mat, label_mat, mean )
	print( weights)
	#display
	display( data_mat, label_mat, weights)
