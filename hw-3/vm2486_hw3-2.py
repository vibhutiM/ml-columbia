#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys 
import os
from numpy.linalg import inv
import itertools

def load_data(directory):
    '''
    this function reads the data from the data folder 
    (directory is same as the location of this script)
    '''
    X_train = np.array(pd.read_csv(directory+'X_train.csv', header = None))
    y_train = np.array(pd.read_csv(directory+'y_train.csv', header = None))
    X_test = np.array(pd.read_csv(directory+'X_test.csv', header = None))
    y_test= np.array(pd.read_csv(directory+'y_test.csv', header = None))
    return X_train, y_train, X_test, y_test


def RBFKernel(x_i, x_j, b=1.0):
    x_i = np.array(x_i)
    x_j = np.array(x_j)
    return np.exp(-(np.sum((x_i-x_j)**2.0))/b)
    
def rmse(y_pred, y_test):
    '''
    calculates rmse given predicted and true valus
    '''
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    return np.sqrt(np.mean((y_pred-y_test)**2))

class GaussianProcess():
    
        
    def fit(self, X, y, sigma_2=1.0, b=1.0):
        n, d= X.shape
        self.K_n = np.zeros((n,n))
        for i in range(n):
            j=i
            while j <n:
                
                self.K_n[i][j]= self.K_n[j][i] = RBFKernel(X[i,:], X[j,:], b)
                j += 1
        self.sigma_2 = sigma_2
        #print self.sigma_2
        
        self.y_train = y
        self.X_train = X
        self.b = b
        #print self.b
    def predict(self, X_test):
        n, d = X_test.shape
        y_pred = np.zeros((n,1))
        for i in range(n):
            K_x_Dn = np.zeros((self.X_train.shape[0],1))
            for j in range(self.X_train.shape[0]):
                K_x_Dn[j] = RBFKernel(X_test[i,:],self.X_train[j,:],self.b)
            
            val_1 = np.dot(inv(self.K_n + (self.sigma_2**2.0 * np.identity(self.X_train.shape[0]))),self.y_train)
            #print val_1
            mu = np.dot(K_x_Dn.T,np.dot(inv(self.K_n + (self.sigma_2* np.identity(self.X_train.shape[0]))),self.y_train))
            #Var = self.sigma_2 + RBFKernel(X_test[i,:],X_test[i,:],b) - np.dot(K_x_Dn,np.dot(inv(self.K_n + (self.sigma_2**2.0 * np.identity(self.X_train.shape[0])),K_x_Dn.T))
            #print mu           
            y_pred[i] = mu
        return y_pred
                
        
            
        
        
    
def prob1(X_train, y_train, X_test, y_test):
    b_arr=np.array([5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    sigma_2_arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    rmse_arr = []
    count = 1
    best_rmse = 1000
    best_pair = [1,1]
    for [b, sigma_2] in list(itertools.product(b_arr, sigma_2_arr)):
        print '...computing for pair no. ', count, ' b: ', b , 'sigma_2: ', sigma_2
        gp = GaussianProcess()
        gp_fit = gp.fit(X_train, y_train, sigma_2, b)
        y_pred=gp.predict(X_test)
        rmse_val = rmse(y_pred, y_test)
        rmse_arr.append([b,sigma_2, rmse_val])
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_pair = [b, sigma_2]
        count += 1
    #print np.array(rmse_arr)
    np.savetxt('rmse_arr.csv', rmse_arr, fmt='%10.4f' , delimiter=',')
    print [best_pair, best_rmse]
    #return np.array(rmse_arr)

def prob1d(X_train, y_train):
    X_train = X_train[:,3]
    n = len(X_train)
    X_train.shape= [n,1]
    
    b  = 5
    sigma_2 = 2
    gp = GaussianProcess()
    gp_fit = gp.fit(X_train, y_train, sigma_2, b)
    y_pred=gp.predict(X_train)
    #print y_pred
    tab = np.zeros((n,3))
    X_train = X_train[:,0]

    for i in range(n):
        tab[i][0] = X_train[i]
        tab[i][1] = y_train[i]
        tab[i][2] = y_pred[i]
    
    
    plt.scatter((tab[:,0]), tab[:,1], label='Scatter Plot', alpha=0.7)
    plt.plot( tab[:,0], tab[:,2], label='Regression Line', linewidth =0.5, color='red')
    
    plt.xlabel('x[4]')

    plt.ylabel('y')
    # and a legend
    plt.legend()
    print '...saving plot from problem 1 (d) as prob1d.png'
    plt.savefig('prob1d.png')


    plt.show()
    plt.close
    
X_train, y_train, X_test, y_test = load_data('/Users/abc/Desktop/spring2017/ML/data/gaussian_process/')
prob1(X_train, y_train, X_test, y_test)
prob1d(X_train, y_train)


def ConfusionMatrix(y1, y2):
    '''
    This function calculates the confusion matrix given true and predicted values. 
    This is also a subroutine of the accyracy function 
    '''
    #print y1.shape, y2.shape
    y_00 = 0
    y_10 = 0
    y_01 = 0
    y_11 = 0
    for i in range(len(y1)):
        if y2[i]==1:
            if y1[i]==1:
                y_11 += 1
            else:
                y_10 += 1
        else:
            if y1[i]==1:
                y_01 += 1
            else:
                y_00 += 1
    result = np.array([y_00,y_01,y_10,y_11])
    result.shape = [2,2]
    return result
    
def accuracy(y1,y2):
    #print 'inhere',y1.shape, y2.shape
    tab = ConfusionMatrix(np.array(y1),np.array(y2))
    return ((1.0*tab[0,0])+tab[1,1])/np.sum(tab)


class Boosting():
    
    def classify(self, y):
        result = np.zeros((len(y),1))
        for i in range(len(y)):
            if y[i] > 0:
                result[i] = 1
            else:
                result[i] = -1
        return result
    def weak_pred(self, X):
        y_pred = np.dot(X,self.w)
        y_pred = self.classify(y_pred)
   
        return y_pred
                    
            
    def err(self,y_pred, y_true):
        n= len(y_pred)
        e_t= 0
        for i in range(n):
            if y_pred[i] != y_true[i]:
                e_t += self.p[i]
                  
        return e_t
    
    def update(self,X, y,t):
        y_pred = self.weak_pred(X)
        e_t = self.err(y_pred, y)
        while e_t>0.5:
            self.w *= -1
            y_pred = self.weak_pred(X)
            e_t = self.err(y_pred, y)
            
        self.exp_val += (0.5-e_t)**2
        self.ub.append([t, np.exp(-2*self.exp_val)])
           
        self.a_t = (0.5 * np.log((1-e_t)/e_t))
        self.eat.append([t, e_t, self.a_t])

        for i in range(len(self.p)):
            self.p[i] *= np.exp(-1.0*y[i]*y_pred[i]*self.a_t)
        den = np.sum(self.p)
        for i in range(len(self.p)):
            self.p[i] = self.p[i]/den

           
        
        
    def LSclassifier(self, X, y):
        self.w = np.dot(inv(np.dot(X.T,X)),np.dot(X.T,y))
        
    def bag(self, X, y):
        n,d= X.shape
        idx = np.random.choice(n, n, replace=True, p= self.p)
        self.prob_baggedvector = self.p[idx]
        for i in range(len(idx)):
            self.freq[idx[i]] += 1
            
        return  X[idx,:] , y[idx]
    
        
    def fit(self, X, y, X_test, y_test):
        sum_alpha_fx_train = np.zeros((X.shape[0],1))
        sum_alpha_fx_test = np.zeros((X_test.shape[0],1))
        
        self.eat = []
        n, d = X.shape
        self.p = np.repeat(1.0/n, n)
        self.decisionplane = np.zeros((d,1))
        np.random.seed(2274)
        T = 1501
        self.tab = []
        self.exp_val = 0
        self.ub = []
        self.freq = {}
        for i in range(n):
            self.freq[i]=0
        for i in range(T):
            print 'iter no: ', i
            #print y_test.shape
            X_bag, y_bag = self.bag(X, y)
            self.LSclassifier(X_bag, y_bag)
            self.update(X, y, i)
            
            y_fx_train = self.weak_pred(X)
            y_fx_test = self.weak_pred(X_test)
          
  
            alpha_fx_train = self.a_t * y_fx_train
            alpha_fx_test = self.a_t * y_fx_test
            
            sum_alpha_fx_train += alpha_fx_train
            sum_alpha_fx_test += alpha_fx_test
            
            
            self.y_train_fit = self.classify(sum_alpha_fx_train)
            self.y_test_pred = self.classify(sum_alpha_fx_test)
      
            
            self.tab.append([i,1.0-accuracy(self.y_train_fit, y), 1.0-accuracy(self.y_test_pred, y_test)])
        self.tab = np.array(self.tab)
    
            

           
    
def prob2(X_train, y_train, X_test, y_test):
    GB = Boosting()
    GB.fit(X_train, y_train, X_test, y_test)
    
    p1, = plt.plot(GB.tab[1:,0], GB.tab[1:,1])
    p2, = plt.plot(GB.tab[1:,0], GB.tab[1:,2])
    
    plt.xlabel('iteration number')
    plt.ylabel('Error')
    plt.legend([p1,p2], ['train error', 'test error'])
    print '...saving plot from problem 2 (a) as prob2a.png'
    plt.savefig('prob2a.png')
    plt.show()
    plt.close()
    
    
    upperbound = np.array(GB.ub)
    plt.plot(upperbound[:,0], upperbound[:,1])
    
    plt.xlabel('iteration number')
    plt.ylabel('Upper Bound Value')
    plt.legend()
    print '...saving plot from problem 2 (b) as prob2b.png'
    plt.savefig('prob2b.png')
    plt.show()
    plt.close()
    
    
    frequency = []
    for i in GB.freq.keys():
        frequency.append([i, GB.freq[i]])
    frequency = np.array(frequency)
    plt.figure(figsize=(5,5))
    plt.bar(frequency[:,0], frequency[:,1], align='center', width=5)
    
    plt.xlabel('Sample no')
    plt.ylabel('Frequency')
    
    print '...saving plot from problem 2 (c) as prob2c.png'
    plt.savefig('prob2c.png')
    plt.show()
    plt.close()

    eat = np.array(GB.eat)
    plt.plot(eat[1:,0], eat[1:,1])
   
   
    plt.xlabel('iteration number')
    plt.ylabel('epsilon')
    
    print '...saving plot from problem 2 (d)1 as prob2d1.png'
    plt.savefig('prob2d1.png')
    plt.show()
    plt.close()
    p2, = plt.plot(eat[1:,0], eat[1:,2])
    plt.xlabel('iteration number')
    plt.ylabel('alpha')
    print '...saving plot from problem 2 (d)2 as prob2d2.png'
    plt.savefig('prob2d2.png')
    plt.show()
    plt.close()
    
    
    
X_train, y_train, X_test, y_test = load_data('/Users/abc/Desktop/spring2017/ML/data/boosting/')
X_train = np.hstack((X_train,np.ones((X_train.shape[0],1))))
X_test = np.hstack((X_test,np.ones((X_test.shape[0],1))))
prob2(X_train, y_train, X_test, y_test)

    
            