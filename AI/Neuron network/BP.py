# coding:utf-8
# Machine learning 西瓜书
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


def preprocess( df ):
    data = df.replace( {'Iris-versicolor':1,'Iris-setosa':0,'Iris-virginica':2} )
    data = data[data['label']<2]
    label = data[['label']]
    datas = data[['sepal_length','speal_width','petal_length','petal_width']]

    return datas , label

logistic = lambda x : 1/(1+np.e**(-1*x))

logistic_derivate = lambda y : y*(1-y)

'''
i_h is a shorthand for input to hidden weights matrix
h_t is a shorthand for input to hidden threshold
h_o is a shorthand for hidden to output weights matrix
o_t is a shorthand for hidden to output threshold
'''
def standard_bp_train( data, label , input_size, inter_size, output_size):
    np.random.seed(100)
    i_h = np.random.rand(inter_size, input_size)
    h_t = np.random.rand( inter_size )
    h_o = np.random.rand(output_size, inter_size)
    o_t = np.random.rand( output_size )
    network = {"hidden_layer":i_h,"hidden_threshold":h_t,
                "output_layer":h_o,"output_threshold": o_t}

    record = []
    step = 0.05
    #BP algorithm
    for _ in range( 40 ):
        for index,row in data.iterrows() :
            y = label.iloc[index,:].values
            x = row.values

            h_input_sum = i_h@x - h_t
            #h_func_ans = np.array( logistic(x) for x in h_input_sum )
            h_func_ans = logistic( h_input_sum )
            #print( h_input_sum, type(h_input_sum), h_input_sum.shape )
            #print( h_func_ans, type(h_func_ans), h_func_ans.shape )

            o_input_sum = h_o@h_func_ans - o_t 
            o_func_ans = logistic( o_input_sum )
            #print( o_func_ans )

            e_wrt_o_input_sum = (o_func_ans-y)* logistic_derivate(o_func_ans)
            #print( "g : " , e_wrt_o_input_sum )
            e_wrt_h_input_sum = e_wrt_o_input_sum@h_o * logistic_derivate(h_func_ans)
            #print( "e : ", e_wrt_h_input_sum , type( e_wrt_h_input_sum ))

            # begin the change
            x = x.reshape(1,x.size)
            e = e_wrt_h_input_sum.reshape(e_wrt_h_input_sum.shape[0],1)
            i_h += -1*step*(e @ x)
            h_t += step*e_wrt_h_input_sum

            temp = h_func_ans.reshape( 1, h_func_ans.size )
            e = e_wrt_o_input_sum.reshape( e_wrt_o_input_sum.size, 1)
            h_o += -1*step*(e@temp)
            o_t += step*e_wrt_o_input_sum
        #correct = test( data, label )
        #record.append( correct )
        right = test( data, label ,network )
        record.append( right )
    return network, record

def fast_bp_train( data, label , input_size, inter_size, output_size):
    np.random.seed(100)
    i_h = np.random.rand(inter_size, input_size)
    h_t = np.random.rand( inter_size )
    h_o = np.random.rand(output_size, inter_size)
    o_t = np.random.rand( output_size )
    network = {"hidden_layer":i_h,"hidden_threshold":h_t,
                "output_layer":h_o,"output_threshold": o_t}

    record = []
    learning_rate = 0.5
    decay_rate = 0.9
    decay_step = 3
    #BP algorithm
    for count in range( 40 ):
        step = learning_rate*decay_rate**(count/decay_step)
        for index,row in data.iterrows() :
            y = label.iloc[index,:].values
            x = row.values

            h_input_sum = i_h@x - h_t
            #h_func_ans = np.array( logistic(x) for x in h_input_sum )
            h_func_ans = logistic( h_input_sum )
            #print( h_input_sum, type(h_input_sum), h_input_sum.shape )
            #print( h_func_ans, type(h_func_ans), h_func_ans.shape )

            o_input_sum = h_o@h_func_ans - o_t 
            o_func_ans = logistic( o_input_sum )
            #print( o_func_ans )

            e_wrt_o_input_sum = (o_func_ans-y)* logistic_derivate(o_func_ans)
            #print( "g : " , e_wrt_o_input_sum )
            e_wrt_h_input_sum = e_wrt_o_input_sum@h_o * logistic_derivate(h_func_ans)
            #print( "e : ", e_wrt_h_input_sum , type( e_wrt_h_input_sum ))

            # begin the change
            x = x.reshape(1,x.size)
            e = e_wrt_h_input_sum.reshape(e_wrt_h_input_sum.shape[0],1)
            i_h += -1*step*(e @ x)
            h_t += step*e_wrt_h_input_sum

            temp = h_func_ans.reshape( 1, h_func_ans.size )
            e = e_wrt_o_input_sum.reshape( e_wrt_o_input_sum.size, 1)
            h_o += -1*step*(e@temp)
            o_t += step*e_wrt_o_input_sum
        #correct = test( data, label )
        #record.append( correct )
        right = test( data, label ,network )
        record.append( right )
    return network,record

'''
    network = {"hidden_layer":i_h,"hidden_threshold":h_t,
                "output_layer":h_o,"output_threshold": o_t}
'''
def predict( row , network ):
    h = network['hidden_layer']@row - network['hidden_threshold']
    h = logistic(h)
    o = network['output_layer']@h - network['output_threshold']
    o = logistic(o)
    return o


def test(data, label , network ):
    right = 0
    for index,row in data.iterrows() :
        x = row.values
        #print( index,  row )
        y_ = predict( x , network )
        #print( y_ )
        #print( label.iloc[index, :].values )
        if list(map(round, y_)) == label.iloc[index,:].values :
            right += 1
    return right 


def display( std, fast ):
    size = std.__len__()
    x = np.linspace(0,size,num=size)
    plt.plot(x, std,'r--', label='std')
    plt.plot(x,fast,'b--',label='fast')
    plt.xlabel('iteration count')
    plt.ylabel('correct prediction')
    plt.legend()
    plt.show()

if __name__=='__main__':
    df = pd.read_csv("iris.data")
    data,label = preprocess( df )
    #print( data )

    input_size = data.shape[1]
    output_size = 1
    inter_size = input_size*2
    
    network, std = standard_bp_train( data,label, input_size, inter_size, output_size)
    print( network )
    right = test( data, label ,network )
    print( right )

    network, fast = fast_bp_train( data,label, input_size, inter_size, output_size)
    print( network )
    right = test( data, label ,network )
    print( right )

    print( std )
    print( fast )
    display( std, fast )

'''    a = data.iloc[1,:].values
    print( a )
    print( a.size , type(a) )

    temp = [0, 1, 0.51, 0.3]
    y = [0,1,1,0]
    b = [0,1,1,0]
    if ( y == b ):
        print('Eqyual')
    c = list(map(round, temp))
    print( c )
    print( 'emmm ')
'''
