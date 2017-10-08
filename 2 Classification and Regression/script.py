import numpy as np
import math as Math
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    unique = np.unique(y)
    k = len(unique)
    d = len(X[0])
    means = np.zeros(shape=(1,3))
    covmat = np.zeros(shape=(d,d))

    fuse = np.append(X,y,axis=1)
    for x in np.nditer(unique):
        A1 = fuse[fuse[:, 2] == x, :]
        means = np.vstack([means,A1.mean(axis=0)])
        np.append(means,A1.mean(axis=0))

    means = np.delete(means, (0), axis=0)
    means = np.delete(means,(d),axis=1)
    means = np.transpose(means)

#     sigma = np.square(np.std(X,axis=0))
#     for x in range(d):
#         covmat[x,x]=sigma[x]

    covmat = np.cov(np.transpose(X))

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    
    unique = np.unique(y)
    k = len(unique)
    d = len(X[0])
    means = np.zeros(shape=(1,3))
    covmats = np.zeros(shape=(k,d,d))

    fuse = np.append(X,y,axis=1)
    for x in np.nditer(unique):
        A1 = fuse[fuse[:, 2] == x, :]
        means = np.vstack([means,A1.mean(axis=0)])
        np.append(means,A1.mean(axis=0))

    means = np.delete(means, (0), axis=0)
    means = np.delete(means,(d),axis=1)
    means = np.transpose(means)

    

#     count=0
#     for x in np.nditer(unique):
#         A1 = fuse[fuse[:, d] == x, :]
#         sigma = np.delete(A1, (d), axis=1)
#         sigma = np.square(np.std(sigma,axis=0))
#         covmat = np.zeros(shape=(d,d))
#         for y in range(d):
#             covmat[y,y]=sigma[y]
#         covmats[count] = covmat
#         count=count+1
  
    
    count=0
    for x in np.nditer(unique):
        A1 = fuse[fuse[:, d] == x, :]
        sigma = np.delete(A1, (d), axis=1)
        covmat = np.cov(np.transpose(sigma))


        covmats[count] = covmat
        count=count+1
    
    
    return means,covmats

def gaussian(mean,covmat,x):
    Dby2 = len(mean)/2
    sqrtsigma = det(covmat)**0.5
    sigmainverse = inv(covmat)
    var = x-mean
    x = 1/(((2*pi)**Dby2)*sqrtsigma)*Math.exp(-0.5*np.dot(np.dot(np.transpose(var),sigmainverse),var))
    return x


def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    means = np.transpose(means)
    i=0
    ypred = np.zeros(shape = (len(Xtest),1))
    for row in Xtest:
        val1 = 0 
        val2 = 1
        count = 1
        for mean in means:
            temp = gaussian(mean,covmat,row)
            if(temp>val1):
                val1=temp
                val2 = count
            count=count+1
        ypred[i] = val2
        i = i+1
    acc = 100*np.mean((ypred == ytest).astype(float))
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    
    means = np.transpose(means)
    i=0
    ypred = np.zeros(shape = (len(Xtest),1))
    for row in Xtest:
        val1 = 0 
        val2 = 1
        count = 1
        for mean in means:
            temp = gaussian(mean,covmats[count-1],row)
            if(temp>val1):
                val1=temp
                val2 = count
            count=count+1
        ypred[i] = val2
        i = i+1
    acc = 100*np.mean((ypred == ytest).astype(float))
    
    
    
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    
    Xtrans = np.transpose(X)
    w = np.dot(np.dot(inv(np.dot(Xtrans,X)),Xtrans),y)
    # IMPLEMENT THIS METHOD  
    
    
    
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD 
    D = len(X[0])
    Xtrans = np.transpose(X)
    w=np.dot(np.dot(inv(lambd*np.identity(D)+np.dot(Xtrans,X)),Xtrans),y)
    
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    
    
    pred = np.dot(Xtest,w)
    mse = np.sum(np.square(pred-ytest))/len(ytest)
    
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    w = w.reshape(len(w),1)
    wtrans = np.transpose(w)
    pred = np.dot(X,w).reshape(len(y),1) 
    deltaL = pred - y
    Xtrans = np.transpose(X)
    error = np.sum(np.square(deltaL))/2 + np.dot(wtrans,w)*lambd*0.5
    error_grad = np.dot(Xtrans,deltaL)+w*lambd
    error_grad = error_grad.flatten()
#     print (error)
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
    
    # IMPLEMENT THIS METHOD
    
    x = x.reshape(len(x),1)
#     print (p)
    if(p==0):
        Xd = np.ones(shape = (len(x),1))
#         print (Xd.shape)
    elif (p == 1):
        Xd = np.ones(shape = (len(x),1))
        Xd = np.append(Xd,x,1)
#         print (Xd.shape)
    else:
        Xd = np.ones(shape = (len(x),1))
        for m in range (1,p+1):
            Xd = np.append(Xd,np.multiply(x,Xd[:,len(Xd[0])-1].reshape(len(x),1)),1)
#         print (Xd.shape)
    
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print ('Problem1:')
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
mle1 = testOLERegression(w,X,y)
w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
mle_i1 = testOLERegression(w_i,X_i,y)
print ('Problem2:')
print('MSE without intercept for training '+str(mle1))
print('MSE with intercept for training '+str(mle_i1))
print('MSE without intercept for test '+str(mle))
print('MSE with intercept for test '+str(mle_i))
l2norm = np.sqrt(np.dot(np.transpose(w_i),w_i))
print('l2norm'+str(l2norm))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
optimum = 0
optimumLambd = 0
optimum1 = 0
optimumLambd1 = 0
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    if(i==0):
        optimum1 = mses3_train[i]
        optimumLambd1 = lambd
        optimum = mses3[i]
        optimumLambd = lambd
    else:
        if(mses3[i]<optimum):
            optimum = mses3[i]
            optimumLambd = lambd
        if(mses3_train[i]<optimum1):
            optimum1 = mses3_train[i]
            optimumLambd1 = lambd
    i = i + 1
print ('Problem3:')
print('Minimum MSE for training data = '+ str(optimum1) + ' for lambda = '+str(optimumLambd1))
print('Minimum MSE for test data = '+ str(optimum) + ' for lambda = '+str(optimumLambd))
w_l = learnRidgeRegression(X_i,y,optimumLambd)
l2norm = np.sqrt(np.dot(np.transpose(w_l),w_l))
print('l2norm'+str(l2norm))
    
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 30}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
goptimum = 0
goptimumLambd = 0
goptimum1 = 0
goptimumLambd1 = 0
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    if(i==0):
        goptimum1 = mses4_train[i]
        goptimumLambd1 = lambd
        goptimum = mses4[i]
        goptimumLambd = lambd
    else:
        if(mses4[i]<optimum):
            goptimum = mses4[i]
            goptimumLambd = lambd
        if(mses4_train[i]<optimum1):
            goptimum1 = mses4_train[i]
            goptimumLambd1 = lambd
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()
print ('Problem4:')
print('Minimum MSE for training data = '+ str(goptimum1) + ' for lambda = '+str(goptimumLambd1))
print('Minimum MSE for test data = '+ str(goptimum) + ' for lambda = '+str(goptimumLambd))

# Problem 5
pmax = 7
lambda_opt = optimumLambd# REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
    
print ('Problem5:')
print (mses5)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()


