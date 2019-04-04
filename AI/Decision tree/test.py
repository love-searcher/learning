###################################
# In CART, it's impossible to have an continue attribute
#
###################################
import numpy as np
import pandas as pd
import copy
import math
import tree

#################################################
# get the attributes information and regular data
#################################################
def preprocess( data ):
	
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
    train_index = [i for i in range(0, data.shape[0],3)]
    valid_index = [i for i in range(1,data.shape[0],3) ] 
    test_index  = [i for i in range(2,data.shape[0],3)]
    train = data.iloc[train_index,:]
    valid = data.iloc[valid_index,:]
    test = data.iloc[test_index,:]
    return data, valid, test
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
	flag = False
	label = most_label( data )

	grouped = data.groupby('label')
	if grouped.size().size == 1 : # all belong to the same class
		label = most_label(data) # get the only label
		flag = True

	if not attributes : # attributes is empty
		flag = True

	data = data.values
	m,n = data.shape
	flag1 = True
	for i in range( n-1 ): # ignore label at -1
		for j in range( m-1 ):
			if data[j][i] != data[j+1][i] :
				flag1 = False  # not finish
				break
	return (flag or flag1), label

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
	for _, group in grouped :
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
	#print(data)
	for attribute in data.columns:
		if ( attribute not in attributes.keys() or attribute == 'label'):
			continue
		if attributes[attribute] : #continue
			gini_index,temp = continue_attribute_gini_index( data, attribute)
		else :
			gini_index = data_attribute_gini_index( data, attribute )
		if gini_index < least_gini_index :
			best_attribute = attribute
			least_gini_index = gini_index
			median = temp  # write bugs all the time
		#print( "attribute: ", attribute,' gini_index:', gini_index )
	return best_attribute, median


###################################################
# generate the tree iteratively.                  #
#                                                 #
###################################################
def build_decision_tree( data, attributes , attributes_value , depth=0):
	root = tree.node()

	flag, label = is_finish(data, attributes) #not empty, no attributes or data can't be seperated
	print( flag, label )
	if flag : # finish 
		root.set_end( label )
		return root

	best_attribute,median = select_best_attribute( data, attributes)
	root.set_attributes(best_attribute, not attributes[best_attribute], median )
	print( best_attribute , ' ', median )
	print( ' ?????'*10 )

	if attributes[best_attribute] == 0 : #distinct
		branch = tree.node()
		grouped = data.groupby( best_attribute )
		for value in attributes_value[best_attribute]:
			if value not in grouped.groups : #数据集中该属性没有这个值 
				branch.set_end( label )
				root.children[value] = branch
				continue
			group = grouped.get_group( value )#get df for the group
			new_attributes = attributes.copy()
			del new_attributes[best_attribute]
			branch = build_decision_tree( group, new_attributes, attributes_value  )
			root.children[value] = branch
	return root

####################################################
# return the number of data with label as 'label'
####################################################
def test( data, label):
	grouped = data.groupby('label')
	if label in grouped.groups :
		group = grouped.get_group( label )
		return group.shape[0]
	return 0
####################################################################
# The select function is in pair with prepruning DT
# 
####################################################################
'''
	select the attribute with most information gain

	correct_num <- ( valid_data, label )
	using the attribute to split the valid_data and data:
		label_ = most_label( data_ )
		correct_num_ += (valid_data_, label_)
	if correct_num_ >= correct_num : #split is better
		return best_attribute, median, True
	return best_attribute, median, False
'''
def select_best_attribute_1( data, attributes,attributes_value, valid_data,label):
	best_attribute,median = select_best_attribute( data, attributes )
	label = most_label( data )
	correct_num = test( valid_data, label)

	data_grouped = data.groupby( best_attribute )
	valid_grouped = valid_data.groupby( best_attribute )
	correct_num_ = 0
	for value in attributes_value[best_attribute] :
		if value not in data_grouped.groups :
			label_ = label
		else :
			label_ = most_label( data_grouped.get_group(value) )

		if value not in valid_grouped.groups : # 
			continue 
		else :
			correct_num_ += test( valid_grouped.get_group(value), label_ )

	if correct_num_ >= correct_num :
		return best_attribute, median, True
	return best_attribute, median, False

####################################################
#  prepruning the tree when build it
#  when split a node, 
# 	if these attributes can't bring more correctness on validation dataset
#		the node will be a leaf
# the valid dataset changes when building the tree
####################################################
def prepruning_DT( data, attributes , attributes_value , valid_data, depth=0):
	root = tree.node()

	flag, label = is_finish(data, attributes) #not empty, no attributes or data can't be seperated
	if flag : # finish 
		root.set_end( label )
		return root

	best_attribute,median,flag = select_best_attribute_1( data, attributes,attributes_value, valid_data,label)
	if (not flag ):# flag means the split is ok
		root.set_end(label)
		return root
	root.set_attributes(best_attribute, not attributes[best_attribute], median )
	#print( best_attribute , ' ', median )

	if attributes[best_attribute] == 0 : #distinct
		branch = tree.node()
		grouped = data.groupby( best_attribute )
		for value in attributes_value[best_attribute]:
			if value not in grouped.groups : #数据集中该属性没有这个值 
				branch.set_end( label )
				root.children[value] = branch
				continue
			group = grouped.get_group( value )#get df for the group
			new_attributes = attributes.copy()
			del new_attributes[best_attribute]
			branch = prepruning_DT( group, new_attributes, attributes_value,valid_data)
			root.children[value] = branch
	return root

####################################################
# give the validation data and the generated tree  
####################################################
def postpruning(root , data , attributes ):
	if ( root.is_a_leaf() ):
		return root

	new_root = tree.node()
	new_root.set_attributes( root.get_attribute(), True)
	branchs = root.get_child_values() #the branchs of this root node
	
	grouped = data.groupby( root.get_attribute() )
	for branch in branchs :
		if branch in grouped.groups : #the data have this attribute value
			child_data = grouped.get_group( branch ) #the data group with attribute values branch
			temp = root.get_child( branch )
			new_child = postpruning( temp , child_data, attributes )
			new_root.set_children(False,branch,new_child)
		else : #the branch has no data to support it.
			new_child = tree.node()
			new_child.set_end( most_label(data) )
			new_root.set_children(False,branch,new_child)


	no_split = tree.node()
	no_split.set_end( most_label(data) )
	precision_pruning = model_test( no_split, data, attributes )

	precision_contrast = model_test( new_root, data, attributes )
	if precision_pruning >= precision_contrast : #original is better
		return no_split
	return new_root

def display( root , depth=0 ):
	if root.is_a_leaf() :
		print( '---'*depth, root.get_label() )
		return
	print( '---'*depth, "<",root.attribute,">",'continue:',root.is_continue )
	for child in root.children :
		print( '---'*depth, child )
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
				print( "Is continue in model test" )
				if item[index] > root.get_median() :
					root = root.get_child('gtr')
				else :
					root = root.get_child('leq')
			else :
				value = item[index]
				root = root.get_child( value )

		if root.get_label() == item[-1] : # correct
			right += 1
		#else :
		#	print( "error: ", item )
	print( "right :",right , " total :" , test_data.shape[0]," correct rate :" , right/test_data.shape[0])
	return right
	#print( data.sort_values( 'density', axis=0) )
	#en, median = continue_attribute_entropy( data, 'density')
	#print( en, "  " ,median )
	#print( most_label(data.iloc[:7,:] ))
	#print( data.values )

if __name__ == '__main__' :
    df = pd.read_csv("car.data")
    data,attributes,attributes_value = preprocess( df )
    train_data, valid_data, test_data = split_dataset( data )	
    #print( train_data )
    #print(test_data )
    print( train_data )
    d_tree = build_decision_tree( train_data , attributes , attributes_value)
    display( d_tree, 1 )
    model_test( d_tree, test_data , attributes )
    model_test( d_tree, data, attributes )
    print( "Postpruning: ","*"*10)
    post_tree = postpruning( d_tree, valid_data, attributes )
    display( post_tree , 1 )
    model_test( post_tree, test_data, attributes )
    model_test( post_tree, data, attributes )
    print( "Prepruning: ","*"*10)
    pre_tree = prepruning_DT( train_data, attributes, attributes_value, valid_data )
    display( pre_tree, 1 )
    model_test( pre_tree, test_data, attributes )
    model_test( pre_tree, data, attributes )
