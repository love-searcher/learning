import logistic_regression as logit
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

###########################################
#logistic_regression used for 2 class.    #
#only get the first two class information #
###########################################
def read_data( file_path ):
	name = ['sepal_length','sepal_width','petal_length','petal_width','class']
	df = pd.read_csv( file_path, header=None, names=name)
	
	m,_ = df.shape
	# convert string to number
	class_mapping = {'Iris-setosa':0,'Iris-versicolor':1}
	df['class'] = df['class'].map( class_mapping )
	df['norm'] = np.ones((m,1))

	#logistic_regression used for 2 class.
	return df[['sepal_length','sepal_width','petal_length','petal_width','norm','class']].values[:100,:]
'''
preprocess data
df.drop(list(range(100,151)), axis=0, inplace=True))
'''
####################################################
# cross_validation                                 #
# using the model algorithm of logistic_regression #
####################################################
def cross_validation( data, k, sep):
	# divide the file into k folders
	dic = {}
	for i in range(k):
		dic[i] = []
	m,_ = data.shape
	for i in range(m):
		dic[i%k].append( data[i] )
	
	# k iteration 
	error_sum = 0
	for i in range(k):
		train = []
		test = []
		for j in range(k):
			if i == j :
				for ele in dic[j] :
					test.append( ele )
			else:
				for ele in dic[j] :
					train.append( ele )
		# divide them into two parts
		train = np.array( train )
		test_set = np.array( test )


		weights = logit.logit_regression( train[:,:sep], train[:,sep:])

		error = logit.test( test_set[:,:sep], test_set[:,sep:], weights )
		error_sum += error
	return error_sum/k

if __name__ =='__main__' :
	data = read_data("iris.data")
	sep = 5 # col 5 is label
	#print( data_mat, label_mat )
	k = 10
	error_rate = cross_validation(data, k, 5)
	print("Cross validation error rate is : ", error_rate)

	k,_ = data.shape
	error_rate = cross_validation(data, k, 5)
	print("Hold one error rate is : ", error_rate)
