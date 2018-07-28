import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv

def euclidean_dist(x1, x2):
    dist = 0.0
    x1 = np.array(x1)
    x2 = np.array(x2)
    x_ = x1 - x2
    for i in range(len(x1)):
        dist += (x_[i]**2)
    return dist

class Kmeans(object):
    def update_c(self, X):
        n, d = X.shape
        for i in range(n):
            dist = 749832658.0
            for j in range(self.n_centers):
                temp_dist= euclidean_dist(X[i],self.centers[j])
                if temp_dist < dist :
                    dist = temp_dist
                    self.class_labels[i] = j
    def update_mu(self, X):
        n, d = X.shape
        n_k = np.zeros((self.n_centers, 1))
        sum_k = np.zeros((self.n_centers,d))
        for i in range(n):
            j = int(self.class_labels[i])
            t = n_k[j]
            t += 1.0
            n_k[j] = t
            t = np.array(sum_k[j])
            t1 = np.array(X[i])
            sum_k[j] = t+t1
            
        for i in range(self.n_centers):
            self.centers[i] = sum_k[i]/n_k[i]
            
    def objective(self, X):
        n, d= X.shape
        L = 0.0
        for i in range(n):
            j = int(self.class_labels[i])
            #print X[i], self.centers[j]
            L += euclidean_dist(X[i], self.centers[j])
        return L
        
        
        
    def fit(self, X, n_centers = 1, n_iter=20):
        self.n_centers = n_centers
        n, d = X.shape
        np.random.seed(434)
        self.centers = np.random.uniform(low=-1,high=1,size=(self.n_centers,d))
        self.class_labels = np.zeros((n,1))
        obj = np.zeros((n_iter, 1))
        for i in range(n_iter):
            self.update_c(X)
            self.update_mu(X)
            obj[i] = self.objective(X)
        return obj
    
def cluster_plot(X, class_labels, centers, k):


    x = X[:,0]
    y = X[:,1]
    Cluster = class_labels   # Labels of cluster 0 to 3


    fig = plt.figure()
    #ax = fig.add_subplot(111)
    scatter = plt.scatter(x,y,c=Cluster, alpha= 0.5)
    for i,j in centers:
        plt.scatter(i,j,c='black',marker='X')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(scatter)
    print '...saving plot from problem 1 (b) as prob1b_'+str(k)+'.png'
    name = 'prob1b_'+ str(k) + '.png'
    plt.savefig(name)   
    plt.show()
    plt.close()
    
def prob1():
    mu_1 = [0, 0]
    sigma_1 = [[1, 0], [0, 1]]
    mu_2 = [3,0]
    sigma_2 = sigma_1
    mu_3 = [0, 3]
    sigma_3 = sigma_1
    data = np.zeros((500,2))
    np.random.seed(325)
    for i in range(500):
        v = np.random.uniform(0,1,1)
        if v <= 0.2:
            data[i] = np.random.multivariate_normal(mu_1, sigma_1, 1)
        elif v <= 0.7:
            data[i] = np.random.multivariate_normal(mu_2, sigma_2, 1)
        else:
            data[i] = np.random.multivariate_normal(mu_3, sigma_3, 1)
            
    K_list = [2, 3, 4, 5]
    for k in K_list:
        clust = Kmeans()
        L = clust.fit(data, n_centers= k, n_iter=20)
        if k == 3:
            cl_3 = clust.class_labels
            cen_3 = clust.centers
        if k == 5:
            cl_5 = clust.class_labels
            cen_5 = clust.centers
            
        x_ = range(20)
        print '...solved for k = ', k
        print 'centroids for k = ', k, 'are : \n', clust.centers
        plt.plot(x_, L, label = 'k = ' + str(k) )
    plt.xlabel('no. of iterations')
    plt.ylabel('Objective function')
    plt.legend()
    print '...saving plot from problem 1 (a) as prob1a.png'
    plt.savefig('prob1a.png')   # save the figure to file
    plt.show()
    plt.close()
    
    cluster_plot(data, cl_3, cen_3, 3)
    cluster_plot(data, cl_5, cen_5, 5)
prob1()   
def load_data():
    '''
    this function reads the data from the data folder 
    (directory is same as the location of this script)
    '''
    data_train = np.array(pd.read_csv('ratings.csv', header = None))
    data_test = np.array(pd.read_csv('ratings_test.csv', header = None))
    movies = (pd.read_table('movies.txt', header = None))
    return data_train, data_test, movies

data_train, data_test, movies = load_data()

def RMSE(y1, y2):
    y1= np.array(y1)
    y2 = np.array(y2)
    y = y1-y2
    return np.sqrt(np.mean(y**2.0))

class MatrixFactorization(object):
    def matrix_norm(self, mat):
        n, d= mat.shape
        res= 0
        for i in range(n):
            for j in range(d):
                res += mat[i][j]**2.0
        return res
                
    
    def log_likelihood(self, sigma_2, lmbd):
        #M_temp = np.dot()
        n_ , d_ = self.M.shape
        M_temp = np.dot(self.U,self.V.T)
        v1 = 0.0
        for i in range(n_):
            for j in range(d_):
                if self.M[i][j] !=0:
                    v1 += (self.M[i][j]-M_temp[i][j])**2
        #v1 = self.matrix_norm(self.M - np.dot(self.U,self.V.T))
        v2 = self.matrix_norm(self.U)
        v3 = self.matrix_norm(self.V)
        res = -1*(((0.5/sigma_2)*v1)+(0.5*lmbd*(self.matrix_norm(self.U)+self.matrix_norm(self.V))))
        return res
        
    def factorise(self, data, d=10, sigma_2 = 0.25, lmbd = 1.0, N_1 = 943, N_2 = 1682, seed= 123):
        n_ , d_ = data.shape
        self.M = np.zeros((N_1, N_2))
        self.U = np.zeros((N_1, d))
        self.V = np.zeros((N_2, d))
        for i in range(n_):
        
            u_ = int(data[i][0])-1
            v_ = int(data[i][1])-1
            r_ = data[i][2]
            self.M[u_][v_] = r_
        np.random.seed(seed)
        mu = np.zeros((d,1))
        mu = mu.T[0]
        Sigma = (1.0/lmbd)*np.eye(d)
        for i in range(N_2):
            self.V[i] = np.random.multivariate_normal(mu, Sigma, 1)
        
        L = np.zeros((99,1))
        for i in range(100):
            print 'iteration number:', i
            for p in range(N_1):
                v1 = lmbd*sigma_2*np.eye(d)
                v2 = np.zeros((10,1))
                for q in range(N_2):
                    if self.M[p][q] != 0:
                        #print p, q
                        #print np.dot(self.V[q],self.V[q].T)
                        t3 = self.V[q,:]
                        t3.shape = ((10,1))
                        #print np.dot(t3, t3.T)
                        v1 += np.dot(t3, t3.T)
                        
                        #self.V[q].shape= (10,1)
                        #print v2.shape, self.V[q].shape
                        a = self.M[p][q]
                        #print v1.shape, self.V[q].shape
                        t1 = np.array(self.V[q])
                        t1.shape = ((10,1))
                        #print t1
                        v2 += a*t1
                        #print v2.shape, v1.shape
                #print self.U[p,:], np.dot(inv(v1), v2)[:,0]
                t2 = np.array(np.dot(inv(v1), v2))
                self.U[p,:] = t2[:,0]
              
            for q in range(N_2):
                v1 = lmbd*sigma_2*np.eye(d) 
                v2 = np.zeros((10,1))
                for p in range(N_1):
                    if self.M[p][q] != 0:
                        #print p,q
                        t3 = self.U[p,:]
                        t3.shape = ((10,1))
                        v1 += np.dot(t3, t3.T)
                        a= self.M[p][q]
                        t1 = np.array(self.U[p])
                        t1.shape = ((10,1))
                        v2 += a*t1
                        #print v1.shape, v2.shape
                        #self.U[p].shape = (10,1)
                        #v2 += self.M[p][q]*self.U[p]
                t2 = np.array(np.dot(inv(v1),v2))
                self.V[q,:] = t2[:,0]
                #print self.V    
            #v2 = v1 + np.dot(self.V.T,self.V)
            #for 
            #self.U = np.dot(inv(v2), np.dot(self.V.T,self.M.T)).T
            #v2 = v1 + np.dot(self.U.T,self.U)
            #self.V = np.dot(inv(v2), np.dot(self.U.T,self.M)).T
            if i > 0:
                L[i-1] = self.log_likelihood(sigma_2, lmbd)
        #return L
        return L, L[98]
            
        
        
def Factorization_predict(data_test, M):
    n , d = data_test.shape
    res = np.zeros((n,1))
    for i in range(n):
        u = int(data_test[i][0])-1
        v = int(data_test[i][1])-1
        res[i] = M[u][v]
    return res

class KNN(object):
    def create_ans(self, movies, k):
        res = []
        for i in range(k):
            m_id = self.idx[i]
            res.append([movies[0][m_id],self.dist[i][0]])
        #print res
        res = np.array(res)
        return res
        
        
    def fit(self, X, movies, j, k = 10):
        n , d = X.shape
        dist = np.zeros((n,1))
        
        for i in range(n):
            dist[i] = np.sqrt(euclidean_dist(X[i],X[j]))
        idx = np.argsort(dist[:,0])
        self.idx = idx[1:(k+1)]
        self.dist = dist[self.idx]
        
        return self.create_ans(movies,k)
    
def prob2(data_train, data_test, mv):
    max_obj = -99999999.0
    tab = np.zeros((10,2))
    x_ = np.arange(2,101)
    for i in range(10):
        #print 'in here'
        # 10
        seed = 123 + (2*i)
        MF = MatrixFactorization()
        L, L_last = MF.factorise(data_train, seed=seed)
        print 'Run no. ', i+1 , 'completed'
        if L_last > max_obj:
            max_obj = L_last
            V = MF.V
            max_i=i+1
        M_iter = np.dot(MF.U, MF.V.T)
        pred_rating = Factorization_predict(data_test, M_iter)
        tab[i][0] = L_last
        tab[i][1] = RMSE(pred_rating, data_test[:,2])
        plt.plot(x_, L, label = 'run no = ' + str(i+1) )
    plt.xlabel('no. of iterations')
    plt.ylabel('Objective function')
    plt.legend()
    print '...saving plot from problem 2 (a) as prob2a.png'
    plt.savefig('prob2a.png')   # save the figure to file
    plt.show()
    plt.close()
    idx = np.argsort(tab[:,0])[::-1]
    tab = tab[idx]
    print tab
    tab = pd.DataFrame(tab)
    tab.to_csv('objective_rmse.csv', sep=',', encoding='utf-8', header=None)
    
    print 'The maximum objective value was obtained from run no: '+ str(max_i)
    
    ######part 2 of problem 2 ######
    q_mv = ['Star Wars (1977)' , 'My Fair Lady (1964)' , 'GoodFellas (1990)']
    
    for i in range(len(q_mv)):
        t1 = mv[mv[0]==q_mv[i]]
        j = t1.index[0]
        nn = KNN()
        t1_ = nn.fit(V, movies, j, k = 10)
        print 'Query movie : ' + q_mv[i] + ' and its 10 nearest neighbors : \n' 
        print t1_
        t1_ = pd.DataFrame(t1_)
        name = 'nn_'+ str(i) +'.csv'
        t1_.to_csv(name, sep=',', encoding='utf-8', header=None)
        
prob2(data_train, data_test, movies)
    
    
    
    
        