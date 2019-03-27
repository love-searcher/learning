###################################################################################
# node as the basic stracture of a tree.
# each one has an attribute
#      it's children is presented as { value : pointer }
#      is_continue is related to the attribute of the node
#            if it's continue, the attribute has a median for decision
#      if the node is leaf, is_leaf will be true
#            and the label means the class it will be or the decision it will make.
###################################################################################
class node():
	def __init__(self):
		self.attribute = ""
		self.is_continue = False  # distinct 
		self.median = None

		self.is_leaf = False # not leaf 
		self.label = None
		self.children = {}
		self.entropy = 0 ; #initial

	def set_attributes( self , name , distinct ,median = None):
		self.attribute = name
		self.is_continue = not(distinct)
		self.median = median

	# set the leaf and children
	def set_children( self , leaf , branch=None,child=None):
		self.is_leaf = leaf 
		self.children[branch] = child  # tree architecture.

	def set_end(self, label ):
		self.is_leaf = True
		self.label = label # the class or the decision


	def is_a_leaf(self ):
		return self.is_leaf

	def get_attribute( self ):
		return self.attribute
	def get_median( self ):
		return self.median
	def get_child( self , branch ):
		if self.is_continue :
			if branch > self.median and 'more' in self.children :
				return self.children['more']
			elif 'less' in self.children :
				return self.children['less']
		#discrete variable
		if branch in self.children:
			return self.children[branch]
		return False  # impossible

	def get_children( self ):
		return self.children.values()
	def get_label(self):
		return self.label
