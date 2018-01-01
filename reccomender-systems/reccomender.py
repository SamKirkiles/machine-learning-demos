# -*- coding: utf-8 -*-

import numpy as np
import pandas
from scipy.optimize import minimize

data = pandas.read_csv('movies.csv').iloc[0:1000,:]
ratings = np.array(pandas.read_csv('ratings.csv'))
ratings = ratings[ratings[:,1] < 1000]

# Preprocessing

print('Preprocessing Data...')

movies = data['genres'].str.split('|')[:,None]
n_labels = np.unique(np.concatenate(movies[:,0]))[1:]


m = movies.shape[0]
n = n_labels.shape[0]

users = np.unique(np.array(ratings[:,0:1]));

r = np.zeros((m,users.shape[0]))

# Set all of the user 

i = 0
for u in users:
    for rating in ratings[ratings[:,0] == u]:
        r[int(rating[1]),i] = rating[2]
    i += 1;


genres = np.zeros((m,n))

for i in range(0,m):
    for j in range(0,n):    
        genres[i,j] = np.in1d(n_labels[j], np.asarray(movies[:,0][i]))
        
print('Done')

# now we have the rating for each movie of each person
# Simple linear regression

def cost(theta, X, y):
    out = np.sum(((X.dot(theta[:,None]) - y)**2)/2)
    grad = (X.dot(theta[:,None]) - y).T.dot(X).flatten()
    return out, grad
    
theta = np.ones(n);

# Set input to someone who only watches crime films
# Model will predict the other half of the crime films as things they should watch
doc = np.zeros(m)
doc[genres[0:500,4] == 1] = 5

out = minimize(cost, x0=theta, args=(genres[doc > 0],doc[:,None][doc > 0]), jac=True)

final = out['x'][:,None].T.dot(genres.T).T