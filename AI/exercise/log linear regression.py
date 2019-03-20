import numpy as np
from matplotlib import pyplot as plt

def log_linear_regression():
	# read data, only the two attributes are useful here.
	data = np.loadtxt("watermelon3.txt")
	x = data[:,-3:-2].copy()
	y = data[:,-2:-1].copy()

	# calculate the corresponding k and bias
	y = np.log(y) # log
	k,b = linear_regression( x, y)

	#display the result
	print(k , type(k))
	print(b)
	plt_result( x,y,k,b)
	
def linear_regression( x, y):
	x_mean = x.sum(axis=0) / x.size
	y_mean = y.sum(axis=0) / y.size
	
	k = (x.T @y + x_mean*y_mean*x.size)/(x.T @x + x.size*x_mean**2)
	bias = y_mean - k*x_mean
	return k[0][0],bias[0][0]   # here, the number is enough

def plt_result( x , y , k , b ):
	x1 = np.linspace( x.min() , x.max() , 100) # attention x.min is a function
	y1 = x1*k+b
	plt.plot( x , y , 'o' ,x1,y1)
	plt.title('log linear regression')
	plt.xlabel('density')
	plt.ylabel('sugar')
	plt.show()
 
	
if __name__ =='__main__':
	log_linear_regression()
