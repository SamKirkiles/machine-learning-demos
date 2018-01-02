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
k = 20
_lambda = 0.5

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

def cost(theta, X, y,l,idx):
    theta = theta.reshape((n,k))
    error = (X.dot(theta) - y) * idx
    out = np.sum((error**2)/2) + ((l/2) * np.sum(theta ** 2))
    grad = error.T.dot(X).T + (l * theta)
    return out, grad.flatten()

theta = np.ones((n,k));
y = r[:,np.random.randint(users.shape[0],size=(k))]

idx = np.zeros((m,k)).astype(bool)
for i in range(0,k):
    idx[:,i] = y[:,i] > 0    


print('Training...')
print('Initial cost: ',cost(theta,genres,y,_lambda,idx)[0])

out = minimize(cost, x0=theta, args=(genres,y, _lambda, idx), jac=True)

print('Done')

print('Final cost: ', out['fun'])
final = out['x'].reshape((n,k)).T.dot(genres.T).T

print('\nUsers Top Rated Movies\n')
print(data.iloc[(np.argmax(y[:,0:1],axis=0))])
print('\nReccomendation\n')
print(data.iloc[(np.argmax(final[:,0:1],axis=0))])
print('\nWouldnt Like\n')
print(data.iloc[(np.argmin(final[:,0:1],axis=0))])

print("Running Collaborative Learning")