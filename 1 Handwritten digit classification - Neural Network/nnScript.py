import pdb
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from math import log
import scipy.misc
import pickle
import pdb
import time




def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1.0 / (1.0 + np.exp(-z))# your code here


def getMask(imgs,mask):
    for i in range(0,len(imgs)):
        img = imgs[i]
        thresh1 = img>=0.7
        mask = mask | thresh1
    return mask
    
def reduceRes(IParr,OParr,OPImgSize):
    for i in range (0,len(IParr)):
        OParr[i]=np.reshape(scipy.misc.imresize(np.reshape(IParr[i],(28,28)),(OPImgSize,OPImgSize)),(1,(OPImgSize*OPImgSize)))
    return OParr
    


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_sample.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     - feature selection"""


    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)

            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.





#preprocess start
    mask = np.zeros(784, dtype=bool)    


    preprocess_method = 0 ## set as 0 for common pixel method, 1 for resolution reduction method
    
    if preprocess_method == 0:    
        new_train_data = np.zeros(train_data.shape)
        new_validation_data = np.zeros(validation_data.shape)
        new_test_data = np.zeros(test_data.shape)
    
        mask = getMask(train_data,mask)     
        new_train_data = np.transpose(np.transpose(train_data)[mask,...])    
        new_validation_data = np.transpose(np.transpose(validation_data)[mask,...])    
        new_test_data = np.transpose(np.transpose(test_data)[mask,...])


    else:
        OPImgSize = 14 # image will be OPImgSize x OPImgSize

        new_train_data = np.zeros((len(train_data),(OPImgSize*OPImgSize)))
        new_validation_data = np.zeros((len(validation_data),(OPImgSize*OPImgSize)))
        new_test_data = np.zeros((len(test_data),(OPImgSize*OPImgSize)))    

        new_train_data = reduceRes(train_data,new_train_data,OPImgSize)
        new_validation_data = reduceRes(validation_data,new_validation_data,OPImgSize)
        new_test_data = reduceRes(test_data,new_test_data,OPImgSize)

    print('preprocess done.. input size reduced from 784 to %i'%(len(new_train_data[1])))

#    pdb.set_trace()
    return new_train_data, train_label, new_validation_data, validation_label, new_test_data, test_label, np.nonzero(mask)

    # print('preprocess done')
    # return train_data, train_label, validation_data, validation_label, test_data, test_label



def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
    # The below code is the forward propagation
    #
    #
    #
    #
    #

    # the entire training data is processed at the same time not one image at a time

    #add a column to training data to represent the bias term

    # print ("forward propagation start")
    new_input_col = np.ones((len(training_data),1))
    training_data = np.append(training_data, new_input_col, 1)


    #calculate the output of the hidden layer
    n_before_hidden_Sigmoid = np.dot(training_data,w1.transpose())
    n_after_sigmoid = sigmoid(n_before_hidden_Sigmoid)
    
    # print ("hidden layer crossed")

    #calculate the output of the output layer
    new_hidden_col = np.ones((len(n_after_sigmoid),1))
    n_after_sigmoid = np.append(n_after_sigmoid,new_hidden_col,1)
    output_before_sigmoid = np.dot(n_after_sigmoid,w2.transpose())
    output_after_sigmoid = sigmoid(output_before_sigmoid)

    # print ("output obtained")

    #   1-deltal   in the formula
    oneminusoutput_after_sigmoid = 1 - output_after_sigmoid
    

    #create actual output(true output) matrix
    Actualoutput = np.zeros((len(training_data), n_class))
    count=0
    for label in training_label:
        Actualoutput[count][int(label)] = 1
        count+=1

    # print("calculation error")

    #these terms are used in the formula to calculate error - obj_val
    oneminusActualOutput = 1 - Actualoutput
    logOutput = np.log(output_after_sigmoid)
    logOneMinusOutput = np.log(oneminusoutput_after_sigmoid)
    mul1 = Actualoutput*logOutput
    mul2 = oneminusActualOutput*logOneMinusOutput
    errorMat = mul1+mul2


    #calculating the obj_val 
    Ji = np.sum(errorMat,1)
    Ji =Ji*-1
    obj_val = np.sum(Ji)/len(training_data)

    # print("calculation error  done")

    #gradiance  calculation for output layer begin
    #delatal   for output layer
    deltaL = output_after_sigmoid - Actualoutput
    #gradiance calculation The vector is initialized to 0's. 
    #in every iteration below gradiance for each input is calculated

    # print("calculating grad_w2 ")


    grad_w2 = np.sum(np.einsum('ij,ik->ijk',deltaL,n_after_sigmoid).reshape(len(training_data),n_class*(n_hidden+1)),0)



    #ignoring the bias term at hidden layer
    n_after_sigmoid = np.delete(n_after_sigmoid, n_hidden, 1)
    w3 = np.delete(w2, n_hidden, 1)

    #grad_w1 begin
    #these terms are used in the formula to calculate error - grad_w1
    oneminus_n_after_sigmoid = 1 - n_after_sigmoid
    zjTerm = oneminus_n_after_sigmoid*n_after_sigmoid
    # dependancyTerm = np.zeros(shape=(1,n_hidden))

    # for delt in deltaL:
    #     dependancyTerm = np.vstack([dependancyTerm, np.dot(delt,w3)])
    # dependancyTerm = np.delete(dependancyTerm, (0), axis=0)
    dependancyTerm = np.einsum('ij,jk->ik',deltaL,w3);


    hiddenTerm = zjTerm*dependancyTerm
    # print("calculating grad_w1 ")


    grad_w1 = np.sum(np.einsum('ij,ik->ijk',hiddenTerm,training_data).reshape(len(training_data),n_hidden*(n_input+1)),0)



    grad_w1=grad_w1+w1.flatten()*lambdaval
    grad_w2 = grad_w2+w2.flatten()*lambdaval


    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad=obj_grad/len(training_data)

    obj_val=obj_val+lambdaval/(2*len(training_data))*(np.sum(np.square(w1))+np.sum(np.square(w2)))

    print ("Erro function value is ...")
    print (obj_val)
    # print ("gradiant ...")
    # print (obj_grad)
    
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels=[]

    #add a column to training data to represent the bias term
    new_input_col = np.ones((len(data),1))
    data = np.append(data, new_input_col, 1)

    #calculate the output of the hidden layer
    n_before_hidden_Sigmoid = np.dot(data,w1.transpose())
    n_after_sigmoid = sigmoid(n_before_hidden_Sigmoid)
    

    #calculate the output of the output layer
    new_hidden_col = np.ones((len(n_after_sigmoid),1))
    n_after_sigmoid = np.append(n_after_sigmoid,new_hidden_col,1)
    output_before_sigmoid = np.dot(n_after_sigmoid,w2.transpose())
    output =[]
    output = sigmoid(output_before_sigmoid)
    for out in output:
        labels = np.append(labels,np.argmax(out))






    # labels = []
    # count =0
    # col = np.ones((len(data),1))
    # data = np.concatenate((data,col),1)
    # for item in data:
    #     #store the sigmoid of dot product of each hidden unit in sigsOfHiddenlayer
    #     sigsOfHiddenlayer =[]

    #     #fIterate over all hidden units of the neural network
    #     for hiddenunit in w1:
    #         sigsOfHiddenlayer = np.append(sigsOfHiddenlayer,[sigmoid(np.dot(hiddenunit.transpose(),data[count]))])

    #     #Add a bias term at the end of the sigsOfHiddenlayer
    #     sigsOfHiddenlayer = np.append(sigsOfHiddenlayer,[1])

    #     #store the sigmoid of dot product of each hidden unit in outputs
    #     output =[]
    #     #fIterate over all output units of the neural network
    #     for outputunit in w2:
    #         output = np.append(output,[sigmoid(np.dot(outputunit.transpose(),sigsOfHiddenlayer))])

    #     labels = np.append(labels,np.argmax(output))
    #     count+=1

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label , selected_features = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 28

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 5


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.


t1 = time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
t2 = time.time()

print ("Time: %s sec"%(t2-t1))
# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)
# predicted_label = nnPredict(initial_w1, initial_w2, train_data)
# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)
# predicted_label = nnPredict(initial_w1, initial_w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)
# predicted_label = nnPredict(initial_w1, initial_w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
obj = [selected_features, n_hidden, w1, w2, lambdaval]
pickle.dump(obj, open('params.pickle', 'wb'))
print ('\n')
