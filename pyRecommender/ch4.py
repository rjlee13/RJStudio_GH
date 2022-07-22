#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:30:16 2022

@author: rj
"""


# set working directory
import os
os.chdir('/Users/rj/Desktop/RJstudio/PyRecommenderSystem')

# =============================================================================
# SGD basic MF ch4.1
# =============================================================================



import numpy as np
import pandas as pd



# Load Data

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('./book_data/u.data', 
                      names    = r_cols,  
                      sep      = '\t',
                      encoding ='latin-1')
ratings.columns # ['user_id', 'movie_id', 'rating', 'timestamp']

# discard timestamp column
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)
ratings

# pivot()
R_temp = ratings.pivot(index   = 'user_id',
                       columns = 'movie_id',
                       values  = 'rating').fillna(0)
R_temp
R_temp.shape  # (943, 1682)


#-------------------------------------------------------

x,y = R_temp.shape 
x # 943
y # 1682

# MF class
class MF():
    def __init__(self, ratings, K, alpha, beta, iterations, verbose = True):
        self.R = np.array(ratings)
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.verbose = verbose

foo = MF(R_temp, K = 30, alpha = 0.001, beta = 0.02, iterations = 100, verbose = True)
foo.R
foo.num_users
foo.num_items
foo.K 
foo.alpha
foo.beta
foo.iterations
foo.verbose

#-------------------------------------------------------



R_temp_np = np.array(R_temp)
R_temp_np.nonzero()

xx = [1,2,3,4]
yy = [1,1,1,1]

for x, y in zip(xx, yy):
    print(x+y)


class MF():
    def __init__(self, ratings, K, alpha, beta, iterations, verbose = True):
        # rating data
        self.R = np.array(ratings)
        # getting number of users and items from ratings
        self.num_users, self.num_items = np.shape(self.R)
        # Number of latent factors
        self.K = K
        # learning rate during update rule
        self.alpha = alpha
        # regularization penalty coef
        self.beta = beta
        # number of iterations when performing SGD
        self.iterations = iterations
        self.verbose = verbose
        
        
        
        # Calculate RMSE (function)
        def rmse(self):
            
            # nonzero element index
            xs, ys = self.R.nonzero()
            
            # initialize predictions and errors as empty lists
            self.predictions = []
            self.errors = []
            
            for x, y in zip(xs,ys):
                prediction = self.get_predict(x, y)
                self.predictions.append(prediction)
                self.errors.append(self.R[x,y] - prediction)
            
            # pupulate 2 lists created above
            self.predictions = np.array(self.predictions)
            self.errors = np.array(self.errors)
            
            # return RMSE
            return np.sqrt(np.mean(self.errors**2))
            

#---------------------
        



np.random.normal(scale = 100, size = (2,3))
np.zeros(3)



class MF():
    def __init__(self, ratings, K, alpha, beta, iterations, verbose = True):
        # rating data
        self.R = np.array(ratings)
        # getting number of users and items from ratings
        self.num_users, self.num_items = np.shape(self.R)
        # Number of latent factors
        self.K = K
        # learning rate during update rule
        self.alpha = alpha
        # regularization penalty coef
        self.beta = beta
        # number of iterations when performing SGD
        self.iterations = iterations
        self.verbose = verbose
        
        
    
    # Calculate RMSE (function)
    def rmse(self):
        
        # nonzero element index
        xs, ys = self.R.nonzero()
        
        # initialize predictions and errors as empty lists
        self.predictions = []
        self.errors = []
        
        for x, y in zip(xs,ys):
            prediction = self.get_prediction(x, y) # defined below
            self.predictions.append(prediction)
            self.errors.append(self.R[x,y] - prediction)
        
        # populate 2 lists created above
        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        
        # return RMSE
        return np.sqrt(np.mean(self.errors**2))


        
    def train(self):
        # random initialization of user & movie latent matrix
        # user 
        self.P = np.random.normal(scale = 1./self.K,
                                  size  = (self.num_users, self.K))
        # movie
        self.Q = np.random.normal(scale = 1./self.K,
                                  size  = (self.num_items, self.K))
        
        # initialize bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()])
        
        
        # training samples
        rows, columns = self.R.nonzero()
        self.samples = [(i,j,self.R[i,j]) for i, j in zip(rows,columns)]
        
        
        # SGD for given number of interations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd() # defined below
            rmse = self.rmse() # defined above
            training_process.append((i+1, rmse))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d / Train RMSE = %.4f" % (i+1, rmse))
        return training_process
        
        
    # rating prediction , r-hat
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i,:].dot(self.Q[j,:].T)
        return prediction
    
    
    # SGD to get optimized P and Q 
    def sgd(self):
        for i,j,r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)
            
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])
            
            self.P[i,:] += self.alpha * (e*self.Q[j,:] - self.beta*self.P[i,:])
            self.Q[j,:] += self.alpha * (e*self.P[i,:] - self.beta*self.Q[j,:])
        
            
mf = MF(R_temp, K = 30, alpha = 0.001, beta = 0.02, iterations = 100, verbose = True)
mf.train()

'''
Iteration: 10 / Train RMSE = 0.9585
Iteration: 20 / Train RMSE = 0.9373
Iteration: 30 / Train RMSE = 0.9280
Iteration: 40 / Train RMSE = 0.9224
Iteration: 50 / Train RMSE = 0.9182
Iteration: 60 / Train RMSE = 0.9142
Iteration: 70 / Train RMSE = 0.9094
Iteration: 80 / Train RMSE = 0.9029
Iteration: 90 / Train RMSE = 0.8936
Iteration: 100 / Train RMSE = 0.8813
'''




# =============================================================================
# train/test MF algorithm ch4.2
# =============================================================================


ratings.shape    # (100000, 3)
ratings.columns  # 'user_id', 'movie_id', 'rating'
len(ratings)     # 100000


from sklearn.utils import shuffle
TRAIN_SIZE = 0.75

ratings_sh = shuffle(ratings, random_state = 1)
ratings_sh # randomly shuffled

cutoff = int(TRAIN_SIZE * len(ratings_sh))
cutoff # 75000

ratings_train = ratings_sh.iloc[:cutoff]
ratings_test  = ratings_sh.iloc[cutoff:]

type(ratings_sh)

for i in ratings_sh:
    print(i) # printing column names

for i, j in enumerate(ratings_sh):
    print(i, j)

R_temp = ratings_sh.pivot(index   = 'user_id',
                          columns = 'movie_id',
                          values  = 'rating').fillna(0)
R_temp.shape # (943, 1682)


for i in R_temp:
    print(i) # printing column ids, movie_id
    
for i,j in enumerate(R_temp):
    print(i,j)
    
boo = []
for i,j in enumerate(R_temp):
    boo.append([i,j])
    break
boo       # [[0, 1]]
dict(boo) #  {0: 1}



class blah():
    def __init__(self, li):
        self.li = li
    def what(self):
        self.haha = []
        for i in range(len(self.li)):
            self.haha.append(i*2)
        return self.haha

x = blah([1,2,3])
x.what()




ratings_test
len(ratings_test) #  25000
for i in ratings_test: 
    print(i)


pow(4-1, 2)

np.newaxis()

# NEW MF class

class NEW_MF():
    def __init__(self, ratings, K, alpha, beta, iterations, verbose = True):
        self.R = np.array(ratings)
        
        item_id_index = []
        index_item_id = []
        for i, one_id in enumerate(ratings):
            item_id_index.append([one_id, i])
            index_item_id.append([i, one_id])
        self.item_id_index = dict(item_id_index)
        self.index_item_id = dict(index_item_id)
        
        user_id_index = []
        index_user_id = []
        for i, one_id in enumerate(ratings.T):
            user_id_index.append([one_id,i])
            index_user_id.append([i,one_id])
        self.user_id_index = dict(user_id_index)
        self.index_user_id = dict(index_user_id)
        
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.verbose = verbose
        
        
    # train set의 RMSE 계산
    def rmse(self):
        xs, ys = self.R.nonzero()
        self.predictions = []
        self.errors = []
        for x, y in zip(xs, ys):
            prediction = self.get_prediction(x, y)
            self.predictions.append(prediction)
            self.errors.append(self.R[x, y] - prediction)
        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors**2))

    # Ratings for user i and item j
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])
    
    
    # Create test set within the class
    def set_test(self, ratings_test):
        test_set = []
        for i in range(len(ratings_test)):
            x = self.user_id_index[ratings_train.iloc[i,0]]
            y = self.item_id_index[ratings_train.iloc[i,1]]
            z = ratings_test.iloc[i,2]
            test_set.append([x,y,z])
            self.R[x,y] = 0
        self.test_set = test_set
        return test_set
    
    # test set RMSE
    def test_rmse(self): 
        error = 0
        for one_set in self.test_set:
            predicted = self.get_prediction(one_set[0], one_set[1])
            error += pow(one_set[2] - predicted, 2)
        return np.sqrt(error / len(self.test_set))
    
    def test(self):
        # Initializing user-feature and item-feature matrix
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()])
        
        rows, columns = self.R.nonzero()
        self.samples = [(i, j, self.R[i,j]) for i, j in zip(rows, columns)]
        training_process = []
        for i in range(1, self.iterations, 6): # MODIFIED for ch4.3
            np.random.shuffle(self.samples)
            self.sgd()
            rmse1 = self.rmse()
            rmse2 = self.test_rmse()
            training_process.append((i+1, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; Train RMSE = %.4f ; Test RMSE = %.4f" % (i+1, rmse1, rmse2))
        return training_process
    
    def get_one_prediction(self, user_id, item_id):
        return self.get_prediction(self.user_id_index[user_id], 
                                   self.item_id_index[item_id])
    
    def full_prediction(self):
        return self.b + self.b_u[:,np.newaxis] + \
            self.b_d[np.newaxis,:] + self.P.dot(self.Q.T)
        

mf = NEW_MF(R_temp, K  = 30, alpha = 0.001, beta = 0.02, iterations = 100)    

test_set = mf.set_test(ratings_test)
result = mf.test()
        
    
"""
Iteration: 10 ; Train RMSE = 0.9681 ; Test RMSE = 1.2121
Iteration: 20 ; Train RMSE = 0.9432 ; Test RMSE = 1.2432
Iteration: 30 ; Train RMSE = 0.9320 ; Test RMSE = 1.2576
Iteration: 40 ; Train RMSE = 0.9252 ; Test RMSE = 1.2662
Iteration: 50 ; Train RMSE = 0.9203 ; Test RMSE = 1.2723
Iteration: 60 ; Train RMSE = 0.9164 ; Test RMSE = 1.2766
Iteration: 70 ; Train RMSE = 0.9125 ; Test RMSE = 1.2800
Iteration: 80 ; Train RMSE = 0.9083 ; Test RMSE = 1.2828
Iteration: 90 ; Train RMSE = 0.9031 ; Test RMSE = 1.2851
Iteration: 100 ; Train RMSE = 0.8963 ; Test RMSE = 1.2872
"""

result

mf.full_prediction()
mf.get_one_prediction(1, 2)




# =============================================================================
# optimal K & iterations ch4.3
# =============================================================================


# optimal K first

results = []
index = []

for K in range(50, 280, 50):
    print('K = ', K)
    mf = NEW_MF(R_temp, K = K, alpha = 0.001, beta = 0.02, 
                iterations = 300)
    test_set = mf.set_test(ratings_test)
    result = mf.test()
    index.append(K)
    results.append(result)

result
len(result) # 50

results
len(results) # 5

for i in results[0]:
    print(i[2])
    break


index


foo = [1,2,3,4]
foo.index(3)

# optimal iterations next

summary = []
for i in range(len(results)):
    RMSE = []
    for result in results[i]:
        RMSE.append(result[2])
    mini = np.min(RMSE)
    j = RMSE.index(mini)
    summary.append([index[i], j, RMSE[j]])

summary















