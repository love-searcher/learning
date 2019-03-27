import numpy as np
#from matploylib import pyplot as plt
import pandas as pd
import copy
import math
import tree

#################################################
# get the attributes information and regular data
#################################################
def preprocess( df ):
	data = df.drop( ['编号','density','ratio_suger'], axis=1)
	
	attributes = {}
	for col in data.columns :
		if( data[col].dtypes == 'float64'):
			attributes[col] = 1
		else :
			attributes[col] = 0

	attributes_value = {}
	m = data.shape[0]
	for col in data.columns :
		if col in attributes.keys() and attributes[col] :# not continue
			continue
		attributes_value[col] = []
		item = data[col].values
		for i in range( m ):
			if item[i] not in attributes_value[col] :
				attributes_value[col].append( item[i] )

	del attributes['label']
	del attributes_value['label']
	return data , attributes, attributes_value

#####################################################
#  split the dataset into train_data and test_data  #
#####################################################
def split_dataset( data ):
	train_index = [1,2,3,6,7,10,14,15,16,17]
	
	test_index = [4,5,8,9,11,12,13]
	train_index = [0,1,2,5,6,9,13,14,15,16]
	test_index = [3,4,7,8,10,11,12]
	train = data.iloc[train_index,:]
	test = data.iloc[test_index,:]
	return train, test
'''
return the most label in data
'''
def most_label( data ):
	labels = data['label'].values
	dic = {}
	for label in labels :
		if label in dic.keys() : #c++ map using count
			dic[label] += 1
		else :
			dic[label] = 1

	big = -1
	ans = None
	for label in dic.keys() :
		if dic[label] > big :
			ans = label
			big = dic[label]
	return ans

'''
if attributes is empty or con't distinguish them
return flag and label : 
	flag is true when it's finished.
	label is the class of the most sample
'''
def is_finish( data, attributes ):
	#most frequency one
	label = most_label( data )
	
	flag = False
	if not attributes : # attributes is empty
		flag = True
	data = data.values
	m,n = data.shape
	for i in range( n-1 ):
		for j in range( m-1 ):
			if data[j][i] != data[j+1][i] :
				flag = False  # not finish
				break
		if flag :
			break
	return flag, label

################################################################
# return the gini of given dataset data #pandas.dataframe      #
################################################################
def get_gini( data ):
	labels = data['label'].values # ndarray
	dic = {}
	for label in labels :
		if label in dic.keys():
			dic[label] += 1
		else :
			dic[label] = 1
	total = labels.shape[0]
	gini = 0
	for label in dic.keys():
		gini += (dic[label]/total)**2
	return 1-gini

######################################################################
# return the weighted gini index of given dataset data and attribute #
# the type of data is pandas.dataframe                               #
######################################################################
def data_attribute_gini_index( data , attribute ):
	grouped = data.groupby(attribute)
	gini_index = 0
	total_size = data.shape[0] # number of sample
	for name, group in grouped :
		temp = get_gini( group )
		gini_index += temp*(group.shape[0]/total_size)
		#print( name , temp )
		#print( entropy )
	return gini_index

###################################################################
# return the best weighted entropy of given dataset and attribute #
# the type of data is pandas.dataframe                            #
# the attribute is continue                                       #
# it find the best sepereate num to get least_gini_index          #
# return  least_gini_index and the corresponding median value     #
###################################################################
def continue_attribute_gini_index( data, attribute):
	data_mat = data.sort_values( attribute )[[attribute,'label']]
	
	least_gini_index = float('inf')
	median = None
	
	size = data_mat.shape[0]
	for i in range( 1,size ):
		front = (i+1)/size * get_gini( data_mat.iloc[:i,:] )
		back = (size-i-1)/size * get_gini( data_mat.iloc[i:,:] )
		gini_index = front + back
		if gini_index < least_gini_index :
			least_gini_index = gini_index
			median = data_mat[attribute].values[i]
	return least_gini_index, median

########################################################################
# return the best attribute given dataset and attributes               #
# if the attribute is continue, return the corresponding median value  #
# pay more attention to the attribute                                  #
########################################################################
def select_best_attribute( data, attributes ):
	best_attribute = None
	least_gini_index = float('inf')
	median = None
	temp = None
	for attribute in attributes.keys():
		if attributes[attribute] : #continue
			gini_index,temp = continue_attribute_gini_index( data, attribute)
		else :
			gini_index = data_attribute_gini_index( data, attribute )
		if gini_index < least_gini_index :
			best_attribute = attribute
			least_gini_index = gini_index
			median = temp  # write bugs all the time
	return best_attribute, median

'''
	print( best_attribute ) # if it is continue
	print( '111111111111' )
	print( '*' * 20 )
	print( depth , ' ', median )
	print( data )
	print( '*' * 20 )
'''
###################################################
# generate the tree iteratively.                  #
#                                                 #
###################################################
def build_decision_tree( data, attributes , attributes_value , depth=0):
	root = tree.node()

	grouped = data.groupby('label')
	if grouped.size().size == 1 : # all belong to the same class
		label = data['label'].values[0] # get the only label
		root.set_end( label )
		return root
	flag, label = is_finish(data, attributes) #not empty
	if flag : # finish 
		root.set_end( label )
		return root
	best_attribute,median = select_best_attribute( data, attributes)
	if attributes[best_attribute] :
		root.set_attributes( best_attribute, False,median )
	else :
		root.set_attributes( best_attribute, True,median )
	if not attributes[best_attribute] : # not distinct
		grouped = data.groupby( best_attribute )
		for value in attributes_value[best_attribute]:
			branch = tree.node()
			if value not in grouped.groups : 
				label = most_label(data)
				branch.set_end( label )
				root.children[value] = branch
				continue
			group = grouped.get_group( value )
			new_attributes = attributes.copy()
			del new_attributes[best_attribute]
			branch = build_decision_tree( group, new_attributes, attributes_value, depth+1 )
			root.children[value] = branch
	else : # continue attribute
#		filter, median
		gtr = data[data[best_attribute]>median]
		leq = data[data[best_attribute]<=median]

		if gtr.shape[0] > 0 :
			new_attributes = attributes.copy()
			del new_attributes[best_attribute]
			branch1 = build_decision_tree( gtr, new_attributes, attributes_value ,depth+1)
			root.children['gtr'] = branch1
		if leq.shape[0] > 0 :
			new_attributes = attributes.copy()
			del new_attributes[best_attribute]
			branch2 = build_decision_tree( leq, new_attributes,attributes_value,depth+1 )
			root.children['leq'] = branch2
	return root

def display( root , depth ):
	if root.is_a_leaf() :
		print( '--'*depth, root.get_label() )
		return
	print( '--'*depth, "<",root.attribute,">" )
	for child in root.children :
		print( '--'*depth, child )
		display( root.children[child], depth+1 )
	return

def model_test( tree , test_data , attributes):
	# 获取attribute在df中的相应列index
	index = 0
	attr_index = {}
	for col in test_data.columns :
		attr_index[col] = index 
		index += 1

	right = 0
	for item in test_data.values :
		root = tree
		while not root.is_a_leaf() :
			attribute = root.get_attribute()
			index = attr_index[attribute]
			if ( attributes[attribute] ): # continue
				if item[index] > root.get_median() :
					root = root.get_child('gtr')
				else :
					root = root.get_child('leq')
			else :
				value = item[index]
				root = root.get_child( value )
		if root.get_label() == item[-1] : # correct
			right += 1
	print( "right :",right , " total :" , test_data.shape[0]," correct rate :" , right/test_data.shape[0])
	#print( data.sort_values( 'density', axis=0) )
	#en, median = continue_attribute_entropy( data, 'density')
	#print( en, "  " ,median )
	#print( most_label(data.iloc[:7,:] ))
	#print( data.values )

if __name__ == '__main__' :
	df = pd.read_csv( "watermelon3.csv")

	data, attributes, attributes_value = preprocess( df )
	train_data, test_data = split_dataset( data )
	print( train_data )
	print(test_data )
	tree = build_decision_tree( train_data , attributes , attributes_value)
	display( tree, 1 )	
	model_test( tree, test_data , attributes )
