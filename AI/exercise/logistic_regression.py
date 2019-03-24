import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_derivative(data_mat, label_mat, weights):
	m,n = data_mat.shape
	p1 = 1/(1+np.exp(-1* data_mat @ weights))
	p0 = 1-p1

	first = (p1-label_mat)*data_mat
	first = first.sum(axis=0)
	first = first.reshape((n,1))
	
	# there are some problem here.
	second = np.zeros((n,n))
	for data,p in zip(data_mat,p1) :
		data = data.reshape((n,1))
		second = second+(data @ data.T)*p*(1-p)
#		print( second )
	return first, second
# using Newton method to calculate the weights
def logit_regression( data_mat, label_mat):
	_,n = data_mat.shape
	weights = np.ones((n,1))

	max_iteration = 500
	for i in range(max_iteration):
		first,second = get_derivative(data_mat, label_mat, weights)
		second = second + 0.01*np.eye(n)
		weights = weights - np.dot(np.linalg.inv(second),first) 
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
	x = np.arange(0.2,0.8,0.1)
	y = -1*(weights[0]*x+weights[2])/weights[1] #两个类的分界0.5不一定合适哦？
	plt.plot( x , y )
	plt.xlabel( 'density' )
	plt.ylabel( 'ratio_suger' )
	plt.show()

def preprocess_data(df):
	m,_ = df.shape
	df['norm'] = np.ones((m,1))
	
	data_mat = df[['density','ratio_suger','norm']].values
	label_mat = df[['label']].values
	return data_mat, label_mat 

	
def test( data , label , weights):
	y = 1/(1+np.exp(-1*data @ weights))
	for i in range(y.size) :
		if y[i] >= 0.5:
			y[i] = 1
		else :
			y[i] = 0
	diff = np.abs(label-y)
	diff = diff.sum( axis = 0)
	return diff/label.size 

if __name__=='__main__':
	# read data from file
	df = pd.read_csv('watermelon3.csv')

	# preprocess data 
	data_mat, label_mat = preprocess_data(df)
#	print( data_mat , data_mat.shape)
#	print( label_mat , type(label_mat), label_mat.shape)
	
	weights = logit_regression( data_mat, label_mat)	
	print( weights )
	error_rate = test(data_mat, label_mat, weights)
	print("error rate is : ", error_rate)
	display( data_mat , label_mat, weights )
