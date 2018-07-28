import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy.linalg as scla

def load_data():
    '''
    this function reads the data from the data folder 
    (directory is same as the location of this script)
    '''
    cfb = np.array(pd.read_csv('CFB2016_scores.csv', header = None))
    names = np.array(pd.read_csv('TeamNames.txt', header = None))
    return cfb, names
    #movies = (pd.read_table('movies.txt', header = None))
    #return data_train, data_test, movies

cfb, names = load_data()

class markov(object):
    def construct_M(self, X):
        self.M = np.zeros((760,760))
        n , d = X.shape
        for row in range(n):
            i = int(X[row][0])-1
            j = int(X[row][2])-1
            vi = (1.0*X[row][1])/(X[row][1] + X[row][3] )
            vj = (1.0*X[row][3])/(X[row][1] + X[row][3] )
            if X[row][1] <= X[row][3]:
                i, j= j, i
                vi, vj = vj, vi
            self.M[i][i] += 1 + vi
            self.M[i][j] += vj
            self.M[j][i] += 1 + vi
            self.M[j][j] += vj
            
                
        for i in range(760):
            self.M[i,:] = self.M[i,:]/np.sum(self.M[i,:])
            
    def calc_stat(self):
        eig_val, eig_vec = LA.eig(np.transpose(self.M))
        idx = np.argsort(eig_val.real)[::-1][:3]
        idx = idx[1]
        u = eig_vec[:,idx].real
        
        
        u.shape = (760,1)
        w_inf = u/np.sum(u)
        print np.sum(w_inf)
        
        return w_inf
     
        
    def part_2(self, i, val):
        
        self.tab[int(i),:] = [i+1, val]
        
    def norm_1 (self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        x = x1 - x2
        res = 0.0
        for i in range(len(x)):
            res += abs(x[i][0])
        return res
    
        
        
        
    
        
    def get_topteam(self, i, names):
        
        a = self.w[:,0]
        idx  = np.argsort(a)[::-1]
        
        idx = idx[0:25]
        
        
        temp = []
        
        for j in range(25):
            temp.append([ names[int(idx[j])],self.w[int(idx[j])] ])
        temp = pd.DataFrame(temp)
        print 'Top 25 teams in iteration no. '+ str(i+1) + ' :'
        print temp
        
        save_name = 'top25_' + str(int(i+1)) + '.csv'
        temp.to_csv(save_name, sep=',', encoding='utf-8', header=None)

    def problem_1(self, X, names):
        self.construct_M(X)
        self.w = np.ones((760,1))/760.0
        
        self.tab = np.zeros((10000,2))
        self.w_inf = self.calc_stat()
        
        for i in range(10000):
            
            self.w = np.dot(self.M.T, self.w)
            
            val = self.norm_1(self.w, self.w_inf)
            
            if i == 9 or i == 99 or i == 999 or i == 9999:
                self.get_topteam(i, names)
            self.tab[i][0] = i+1
            self.tab[i][1] = val
            self.part_2(i,val)
        plt.plot(self.tab[:,0], self.tab[:,1])
        plt.xlabel('Iteration Number')
        plt.ylabel('|w_inf - w(t)|')
        figure= plt.gcf()
        figure.set_size_inches(10,6)
        print '...saving plot for problem 1 as prob1.png'
        plt.savefig('prob1.png')

        plt.show()
        plt.close()
        
mc = markov()
mc.problem_1(cfb, names)


##########################
######problem 2 ##########
##########################

nyt = pd.read_csv('nyt_data.txt', sep='\n', header= None)
X = np.zeros((3012,8447))
for i in range(8447):
    temp = nyt[0][i].split(',')
    for j in range(len(temp)):
        idx, ct = temp[j].split(':')
        idx = int(idx)-1
        ct = int(ct)
        X[idx][i] = ct
np.random.seed(3213)
n1, n2 = X.shape
W = np.random.uniform(low=0.1,high=2,size=(n1,25))
H = np.random.uniform(low=0.1, high=2, size=(25,n2))

def divergence(X, W, H):
    n1, n2 = X.shape
    return np.sum(-np.multiply(X, np.log(np.dot(W,H)+(10**-16)*np.ones((n1, n2))))+np.dot(W,H))
        
def update(X,W,H):
    n1, n2 = X.shape 
    p= np.divide(X, (np.dot(W, H)+(10**-16)*np.ones((n1, n2))))
    normal=W/(W.sum(axis=0)+10**-16)
    tmp= np.dot(np.transpose(normal),p)
    H= np.multiply(H,tmp)
    p= np.divide(X, (np.dot(W, H)+(10**-16)*np.ones((n1, n2))))
    normal=np.transpose(np.transpose(H)/(np.transpose(H).sum(axis=0)+10**-16))
    tmp= np.dot(p,np.transpose(normal))
    W= np.multiply(W,tmp)
    return H, W
tab = []
for i in range(100):
    print i
    H, W = update(X,W,H)
    tab.append([i+1,divergence(X, W, H)])
#print tab
tab = np.array(tab)
print tab
plt.plot(tab[:,0], tab[:,1])
plt.xlabel('iteration number')
plt.ylabel('objective value')
plt.savefig('prob2.png')
plt.show()
plt.close()

words = np.array(pd.read_csv('nyt_vocab.dat', header= None))
for i in range(25):
    arr = W[:,i]
    val = np.sum(arr)*1.0
    arr = arr/val
    
    idx = np.argsort(arr)[::-1][:10]
    print idx
    temp =[]
    for id in idx:
        temp.append([words[id][0],arr[id]])
    #print temp
    temp = np.array(temp)
    print 'Column ', i+1, ' : '
    print temp
    save_name = 'top10words_' + str(int(i+1)) + '.csv'
    temp = pd.DataFrame(temp)
    temp.to_csv(save_name, sep=',', encoding='utf-8', header=None)
        