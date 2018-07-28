import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import sys 
import os
import numpy.linalg as lin
import scipy.special as sp

def ConfusionMatrix(y_pred, y):
    '''
    This function calculates the confusion matrix given true and predicted values. 
    This is also a subroutine of the accyracy function 
    '''
    y_00 = 0
    y_10 = 0
    y_01 = 0
    y_11 = 0
    for i in range(len(y)):
        if y[i]==1:
            if y_pred[i]==1:
                y_11 += 1
            else:
                y_10 += 1
        else:
            if y_pred[i]==1:
                y_01 += 1
            else:
                y_00 += 1
    result = np.array([y_00,y_01,y_10,y_11])
    result.shape = [2,2]
    return result
    
def accuracy(y_pred,y_test):
    tab = ConfusionMatrix(np.array(y_pred),np.array(y_test))
    return ((1.0*tab[0,0])+tab[1,1])/np.sum(tab)





class NaiveBayes(object):
    '''
    The class NaiveBayes fits a naive bayes classifier object on the input data.
    NOTE: Need to input an indicator variable which decides the conditional probability distribution of the particular feature 
    : 1 is for bernaulli distn and 2 is for pareto distn.
   
    X and y are numpy arrays of appripriate dimensions.
    
    This class has 2 modules fit and predict to perform fitting of training 
    data and prediction for test data respectively.
    '''

    def fit(self, X, y, indicator):
        theta_0 = []
        theta_1 = []
        self.idx_y0 = []
        self.idx_y1 = []
        
        for i in range(len(y)):
            if y[i]==0:
                self.idx_y0.append(i)
            else:
                self.idx_y1.append(i)
        n,d = X.shape 
        for i in range(d):
            if indicator[i]==1:
                temp_0 = np.mean(X[self.idx_y0,i])
                temp_1 = np.mean(X[self.idx_y1,i])
            else:
                temp_0 = 1.0/(np.mean(np.log(X[self.idx_y0,i])))
                temp_1 = 1.0/(np.mean(np.log(X[self.idx_y1,i])))
            theta_0.append(temp_0)
            theta_1.append(temp_1)
        
        self.theta0 = np.array(theta_0)
        self.theta1 = np.array(theta_1)
        self.pi = np.mean(y) 
        self.indicator = indicator
    
    def predict(self, X):
        n,d = X.shape 
        pred = []
        
        for i in range(n):
            pred0 = 1
            pred1 = 1
            for j in range(d):
                if self.indicator[j]==1:
                    pred0 *=  (self.theta0[j]**X[i,j])*((1-self.theta0[j])**(1-X[i,j]))
                    pred1 *=  (self.theta1[j]**X[i,j])*((1-self.theta1[j])**(1-X[i,j]))
                if self.indicator[j]==2:
                    pred0 *= self.theta0[j]*X[i,j]**(-(self.theta0[j]+1))
                    pred1 *= self.theta1[j]*X[i,j]**(-(self.theta1[j]+1))
            y_1 = self.pi*pred1
            y_0 = (1-self.pi)*pred0
            if y_1 > y_0:
                pred.append(1)
            else:
                pred.append(0)
        pred = np.array(pred)
            
        return pred
    
class knn(object):
    '''
    KNN classifier. Provide 'k' as a list object. 
    
    This class has 2 modules fit and predict to perform fitting of training 
    data and prediction for test data respectively.
    '''
    def fit(self, X, y, k=[1]):
        self.X_train = X
        self.y_train = y
        self.k = k
        
    def majority(self,a,k):
        res = 0
        for i in range(k):
            res += a[i]
        if res <= k/2.0:
            return 0
        else:
            return 1
        
    def sort_y(self,a1,a2):
        
        idx=np.argsort(a1)
        a2 = a2[idx]
        return a2
            
            
    def l1_distance(self,a,b):
        res = 0
        for i in range(len(a)):
            res += abs(a[i]-b[i])
        return res
            
    def predict(self,X_test):
        n,d =X_test.shape
        N, D = self.X_train.shape
        y_pred = []
        for i in range(n):
            temp = []
            for k in range(N):
                dist = self.l1_distance(self.X_train[k,:],X_test[i,:])
                temp.append(dist)
            a1 = np.array(temp)
            a2 = np.array(self.y_train)
            
            
            arranged_y = self.sort_y(a1,a2)
            vote = []
            for val in self.k:
                vote.append(self.majority(arranged_y,val))
            y_pred.append(vote)
        y_pred = np.array(y_pred)
        self.y_pred = y_pred
        return y_pred

    
class LogisticRegression(object):
    '''Logistic regression classifier. The input X should include an offset coloumn (a coloumn of 1)
    X and y should be numpy arrays of appropriate dimensions.. 
    '''
 
    
    def sigmoid(self, x, y, w):
        
        
        power = np.dot(x.T,w)
        #print power
        #print power.shape
        #if power > 200:
            #if y == 1:
                #return 0.9999
            #else:
                #return 0.0001
        calc = sp.expit(power)
        if calc == 1:
            calc = 0.9999
        if calc == 0:
            calc = 0.0001
        if y ==1 :
            return calc
        else:
            return 1.0 - calc
        
    def grad(self, X, y, w):
        n, d = X.shape
        result = np.zeros((d,1))
        #print 'result', result
        for i in range(n):
            #print w.shape
            x= X[i,:]
            x.shape = [len(x),1]
            #print x.shape
            constant = (1.0-self.sigmoid(x,y[i],w))*y[i]
            result += (constant*x)
        return result
    
    def objective(self,X, y, w):
        n, d = X.shape
        #print len(w)
        result = 0
        for i in range(n):
            x= X[i,:]
            x.shape = [len(x),1]
            constant = self.sigmoid(x,y[i],w)
            if constant == 0:
                constant = 0.0001
            #print constant
            result += math.log(constant)
        return result
            
    def fit(self, X, y):
        n,d = X.shape
        w = np.zeros((d,1))
        L = []
        for i in range(10000):
            if i%100 == 0:
                print 'iteration ',i
            neta = 1.0/(10**5*math.sqrt(i+1.0))
            temp_L = [i,self.objective(X,y,w)]
            L.append(temp_L)
            val = neta*self.grad(X,y,w)
            val.shape = [len(val),1]
            w.shape = [len(w),1]
            w = w+val
        self.w = w
        self.w.shape = [len(self.w),1]
        self.L = np.array(L)
        
    def predict(self, X):
        n,d = X.shape
        result = []
        for i in range(n):
            x= X[i,:]
            x.shape = [len(x),1]
            val = np.dot(x.T,self.w)

            if val >= 0:
                result.append(1)
            else:
                result.append(-1)
        return np.array(result)
        
class NewtonMethod(object):
    '''Newton's method  classifier. The input X should include an offset coloumn (a coloumn of 1)
    X and y should be numpy arrays of appropriate dimensions.. 
    '''
   
    
    def sigmoid(self, x, y, w):
        #x = np.array(x)
        #w = np.array(w)
        
        power = np.dot(x.T,w)
        #print power
        #print power.shape
         
        calc = sp.expit(power)
        if calc ==1:
            calc = 0.9999
        if calc ==0:
            calc = 0.001
        if y ==1 :
            return calc
        else:
            return 1.0 - calc
    def grad(self, X, y, w):
        n, d = X.shape
        result = np.zeros((d,1))
        #print 'result', result
        for i in range(n):
            #print w.shape
            x= X[i,:]
            x.shape = [len(x),1]
            #print x.shape
            constant = (1.0-self.sigmoid(x,y[i],w))*y[i]
            result += constant*x
        return result
    
    def objective(self,X, y, w):
        n, d = X.shape
        #print len(w)
        result = 0
        for i in range(n):
            x= X[i,:]
            x.shape = [len(x),1]
            constant = self.sigmoid(x,y[i],w)
            if constant == 0:
                constant = 0.0001
            #print constant
            result += math.log(constant)
        return result
    
    def hessian(self, X, y,w):
        n,d = X.shape
        result = np.zeros([d,d])
        for i in range(n):
            x= X[i,:]
            x.shape = [len(x),1]
            constant = self.sigmoid(x,y[i],w)
            constant = constant * (1.0-constant)
            result += constant*np.dot(x,x.T)
        #print lin.det(result)
        result *= -1.0
        if lin.det(result)== 0.0:
            lambda_I = 0.0001*np.identity(x.shape[0])
            result += lambda_I
            #result = lin.inv(result)
        #else:
        result = lin.inv(result)
            #self.hess = result
        return result
        
            
    def fit(self, X, y):
        n,d = X.shape
        w = np.zeros((d,1))
        L = []
        for i in range(100):
            #print i
            neta = 1.0/(math.sqrt(i+1.0))
            temp_L = [i,self.objective(X,y,w)]
            L.append(temp_L)
            gradient  = self.grad(X,y,w)
            gradient.shape = [len(gradient),1]
            inv_hessian = self.hessian(X,y,w)
            val = np.dot(inv_hessian,gradient)
            val *= (-1.0*neta)
            w += val
            w.shape = [len(w),1]
            val.shape = [len(val),1]
            w = w+val
            
        self.w = w
        self.w.shape = [len(w),1]
        self.L = np.array(L)
        
    def predict(self, X):
        n,d = X.shape
        result = []
        for i in range(n):
            x= X[i,:]
            x.shape = [len(x),1]
            val = np.dot(x.T,self.w)
            if val >= 0 :
                result.append(1)
            else:
                result.append(-1)
        return np.array(result)

    

                
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



def prob_2a(X_train,y_train,X_test,y_test):
    n,d= X_train.shape
    indicator = []
    for i in range(d):
        if i <54:
            indicator.append(1)
        else:
            indicator.append(2)
    indicator = np.array(indicator)
    indicator
    model = NaiveBayes()
    fit1 = model.fit(X_train,y_train,indicator)
    y_pred=model.predict(X_test)
    print 'Confusion Matrix for naive bayes classifier: ', ConfusionMatrix(y_pred,y_test)
    print 'accuracy:', accuracy(y_pred, y_test)
    #len(model.theta0[0:54])
    x=range(1,55)
    markerline, stemlines, baseline = plt.stem(x, model.theta0[0:54],  markerfmt='o',label = 'theta_1 for y=0')
    plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    plt.setp(stemlines, 'linestyle', 'dotted')


    markerline, stemlines, baseline = plt.stem(x,model.theta1[0:54], markerfmt='o', label='theta_1 for y=1')
    plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    plt.setp(stemlines, 'linestyle', 'dotted')

    #plt.legend()
    #plt.show()
    
   
    plt.legend()
    print '...saving plot from problem 2 (b) as prob2b.png'
    plt.savefig('prob2b.png')
    plt.show()
    plt.close()

def prob_2c(X_train,y_train,X_test,y_test):
    k = []
    for i in range(21):
        k.append(i)
    clf = knn()
    clf.fit(X_train,y_train,k)
    pred_mat = clf.predict(X_test)
    result = []
    for i in range(len(k)) :
        
        pred_y = pred_mat[:,i]
        temp = [k[i],accuracy(pred_y,y_test)]
        result.append(temp)
    result = np.array(result)
    
    plt.plot(result[:,0], result[:,1])
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of knn for different values of k')
    print '...saving plot from problem 2 (c) as prob2c.png'
    plt.savefig('prob2c.png')
    plt.show()
    plt.close()
    print 'Accuracy of knn for different values of k: ', result

def prob_2d(X_train, y_train, X_test, y_test):
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    print 'in here'
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    for i in range(len(y_train)):
        if y_train[i]==0:
            y_train[i] = -1
    for i in range(len(y_test)):
        if y_test[i]==0:
            y_test[i] = -1  
    logclf = LogisticRegression()
    fit = logclf.fit(X_train, y_train)
    pred_y = logclf.predict(X_test)
    
    plt.plot(logclf.L[:,0], logclf.L[:,1])
    plt.xlabel('t')
    plt.ylabel('Objective Function (L)')
    print '...saving plot from problem 2 (d) as prob2d.png'
    plt.savefig('prob2d.png')
    plt.show()
    plt.close()
    print 'Confusion Matrix for Logistic Regression:', ConfusionMatrix(pred_y,y_test)
    
def prob_2e(X_train, y_train, X_test, y_test):
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    for i in range(len(y_train)):
        if y_train[i]==0:
            y_train[i] = -1
    for i in range(len(y_test)):
        if y_test[i]==0:
            y_test[i] = -1 
    for i in range(len(y_test)):
        if y_test[i]==0:
            y_test[i] = -1  
    nmclf = NewtonMethod()
    fit = nmclf.fit(X_train, y_train)
    pred_y = nmclf.predict(X_test)
    
    
    plt.plot(nmclf.L[:,0], nmclf.L[:,1])
    plt.xlabel('t')
    plt.ylabel('Objective Function (L)')
    print '...saving plot from problem 2 (e) as prob2e.png'
    plt.savefig('prob2e.png')
    plt.show()
    plt.close()
    print 'Confusion Matrix:', ConfusionMatrix(pred_y,y_test)
    print 'accuracy of newtons method:', accuracy(pred_y, y_test)


X_train, y_train, X_test, y_test = load_data()
prob_2a(X_train,y_train,X_test,y_test)
prob_2c(X_train,y_train,X_test,y_test)
prob_2e(X_train,y_train,X_test,y_test)
prob_2d(X_train,y_train,X_test,y_test)