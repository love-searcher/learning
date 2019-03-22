import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

'''
using gradient ascent to converge
'''
def grad_ascent( data_mat , label_mat ):
	m,n = data_mat.shape
	
	alpha = 0.1
	max_iteration = 500
	weights = np.ones((n,1))

	for k in range( max_iteration ):
		a = data_mat @ weights # (17,3)*(3,1)
		h = 1/(1+np.exp(-1*a)) # (17,1)
		error = (label_mat-h) # (17,1)
		weights = weights+alpha*(data_mat.T @ error)  #this is the corresponding derivative
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
	y = -1*(weights[0]*x+weights[2]-0.5)/weights[1]
	plt.plot( x , y )
	plt.xlabel( 'density' )
	plt.ylabel( 'ratio_suger' )
	plt.show()
	
if __name__=='__main__':
	#chinese, we need to pay more attention to the encode, UTF-8
	df = pd.read_csv("watermelon3.csv")
	m,n = df.shape
	# append a column for bias
	df['norm'] = np.ones((m,1))

	data_mat = df[['density','ratio_suger','norm']].values
	label_mat = df[['label']].values # return type is np.ndarray
#	label_mat = label_mat.T 
#	print( data_mat , type(data_mat))
#	print( label_mat, type(label_mat))
	weights = grad_ascent( data_mat , label_mat )
	print( weights )
	display( data_mat, label_mat, weights )
