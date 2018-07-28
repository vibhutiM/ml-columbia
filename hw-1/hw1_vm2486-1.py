#from __future__ import __main__
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys 
import os

class Ridge(object):
    '''
    The class ridge performs ridge regression where the input value X has
    an already appended coloumn of 1 to account for bias in the regressed line.
    X and y are numpy arrays of appripriate dimensions.
    
    This class has 2 modules fit and predict to perform fitting of training 
    data and prediction for test data respectively.
    '''

    def fit(self, X, y, lmbd=0):
        n,d = X.shape
        
        lI = lmbd * np.eye(d)
        self.w = np.dot(np.linalg.inv(np.dot(X.T, X) + lI),
                             np.dot(X.T, y))

    def predict(self, X):
        return np.dot(X, self.w)
    
    
def dof(X, lmbd=0):
    '''
    Note : This function takes the diagonal matrix from singular value distribution of the 
    input values X and not X itself. I did this to reduce the number of computations since 
    SVD won't change for different values of lambda.
    
    s is a dxd diagonal matrix (numpy array)
     
    '''
    U, s, v = np.linalg.svd(X)
    df = 0
    for i in s :
        df +=  i**2.0/(lmbd+i**2.0)
        #(math.pow(i,2))/(lmbd+math.pow(i,2))
    return df


def prob_1a(X, y):
    '''
    Solution code for 1(a).
    X and y are numpy arrays
    '''
    #U, s, v = np.linalg.svd(X)
    table = []
    for lmbd in range(0,5001):
        r= Ridge()
        r.fit(X,y,lmbd=lmbd)
        row = list(r.w)     #using list for dynamic allocation
        row.append(dof(X,lmbd))
        table.append(row)
    table = np.array(table) #converting back to numpy array for further manipulation
    
    
    '''
        table is a numpy array of where the last coloumn is df(lambda) [for lambda ranging from 0 to 5000 ]
        the previous columns are the regression variables: w1, w2, w3,...,w7 (in this particular case)
    '''
        
    for i in range((table.shape[1])-1):
        plt.plot( table[:,table.shape[1]-1], table[:,i], label = 'feature' + str(i+1))
    plt.xlabel('df(lambda)')
    
    plt.legend()
    print '...saving plot from problem 1 (a) as prob1a.png'
    plt.savefig('prob1a.png')
    plt.show()
    plt.close()

    
def rmse(y_pred, y_test):
    '''
    calculates rmse given predicted and true valus
    '''
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    return np.sqrt(np.mean((y_pred-y_test)**2))


def prob_1c(X, y, X_test, y_test):
    '''
    Solution code for 1(c).
    X and y are numpy arrays
    '''
    table = []
    for lmbd in range(0,51):
        r= Ridge()
        r.fit(X,y,lmbd=lmbd)
        pred = r.predict(X_test)
        rmse_sq = rmse(pred,y_test)
        temp = [lmbd,rmse_sq]
        table.append(temp)
    table = np.array(table)
    
    '''
        table is a numpy array of size 51x2 
        Coloumn 1 has values of lambda
        Coloumn 2 has respectuve RMSE values 
    '''
        
        
    plt.plot(table[:,0], table[:,1])
    plt.xlabel('lambda')
    plt.ylabel('RMSE')
    print '...saving plot from problem 1 (c) as prob1c.png'
    plt.savefig('prob1c.png')
    plt.show()
    plt.close()
    

def make_2d(X):
    '''
    This function converts an input numpy array for regression of degree 1 to an input for 
    regression of degree 2 (without cross terms)
    NOTE: X already has the coloumns for 1 at the end
    '''
    
    X_2 = X
    for i in range(X.shape[1]-1):
        X_2 = np.hstack((X_2, np.power(np.array(X[:,i]).reshape(X.shape[0],1), 2)))
    return X_2

def make_3d(X):
    '''
    This function converts an input numpy array for regression of degree 1 to an input for 
    regression of degree 3 (without cross terms)
    NOTE: X already has the coloumns for 1 at the end
    '''
    
    X_3 = X
    for i in range(X.shape[1]-1):
        X_3 = np.hstack((X_3, np.power(np.array(X[:,i]).reshape(X.shape[0],1), 2), 
                         np.power(np.array(X[:,i]).reshape(X.shape[0],1),3)))
    return X_3


def rmse_gen(X, y, X_test, y_test, l_lim=0, u_lim=500):
    '''
    returns RMSE values for different values of lambda in a table 
    '''
    result = []
    for i in range(l_lim,u_lim+1):
        r = Ridge()
        r.fit(X,y, lmbd=i)
        pred = r.predict(X_test)
        result.append(rmse(pred,y_test))
    return np.array(result)



def prob_2d(X, y, X_test, y_test):
    
    '''
    Solution code for 2(d).
    X and y are numpy arrays
    '''
    
    c = np.arange(501)
    table_1 = rmse_gen(X, y, X_test, y_test)
    table_2 = rmse_gen(make_2d(X), y, make_2d(X_test), y_test)
    table_3 = rmse_gen(make_3d(X), y , make_3d(X_test), y_test)
    fig = plt.plot(c, table_1, label = 'p=1')
    fig = plt.plot(c, table_2, label = 'p=2')
    fig = plt.plot(c, table_3, label = 'p=3')
    plt.xlabel('lambda')
    plt.ylabel('RMSE')
    plt.legend()
    print '...saving plot from problem 2 (d) as prob2d.png'
    plt.savefig('prob2d.png')   # save the figure to file
    plt.show()
    plt.close()
    print 'for p=1, lambda =', list(table_1).index(min(table_1)), 'gives the minimum RMSE of ', min(table_1)
    print 'for p=2, lambda =', list(table_2).index(min(table_2)), 'gives the minimum RMSE of ', min(table_2)
    print 'for p=1, lambda =', list(table_3).index(min(table_3)), 'gives the minimum RMSE of ', min(table_3)

def load_data():
    '''
    this function reads the data from the data folder 
    (directory is same as the location of this script)
    '''
    X_train = np.array(pd.read_csv('X_train.csv', header = None))
    y_train = np.array(pd.read_csv('y_train.csv', header = None))
    X_test = np.array(pd.read_csv('X_test.csv', header = None))
    y_test= np.array(pd.read_csv('y_test.csv', header = None))
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()
  
prob_1a(X_train, y_train)
prob_1c(X_train, y_train, X_test, y_test)
prob_2d(X_train, y_train, X_test, y_test)

  