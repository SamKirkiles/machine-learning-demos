# -*- coding: utf-8 -*-

from sklearn.datasets import make_classification;
import matplotlib.pyplot as plt;
import numpy as np;
from cvxopt.solvers import qp;
import cvxopt.solvers;

# Langerian svm on data set without kernels

# Create training data with two features and binary classification
X,y = make_classification(n_samples=100, n_redundant=0, class_sep=1.5, n_informative=1, n_features=2, n_clusters_per_class=1, random_state=345);

y[y==0] = -1
hard = False;
C = 0.5;

#Show a scatter plot of our data
plt.scatter(X[:,0][np.where(y == 1.0)], X[:,1][np.where(y == 1.0)], marker='o', c='#4e71c9');
plt.scatter(X[:,0][np.where(y == -1.0)], X[:,1][np.where(y == -1.0)], marker='o', c='#b53636');

m = X.shape[0];
y = np.reshape(y,(m,1));

# We must find the extrema of the function in the form 1/2*a'Pa-q'x subject to constraints


Py = y.dot(y.T);
Px = X.dot(X.T);
P = cvxopt.matrix(Py * Px);

q = cvxopt.matrix(-1.0 * np.ones((m)));

if(hard):
    G = cvxopt.matrix(np.diag(-1.0 * np.ones((m))));
    h = cvxopt.matrix(np.zeros((m)));
else:
    G = cvxopt.matrix(np.vstack((np.eye(m) * -1, np.eye(m))));
    inner = np.hstack((np.zeros(m), np.ones(m) * C))
    h = cvxopt.matrix(inner);

A = cvxopt.matrix(y.reshape(1,m).astype(float));
b = cvxopt.matrix(0.0);

solution = qp(P,q, G, h, A, b);

multipliers = np.ravel(solution['x']);

# We need to only use the support vectors 
# Why are there only negative sample support vectors?
has_positive_multiplier = multipliers > 1e-7
sv_multipliers = multipliers[has_positive_multiplier]
support_vectors = X[has_positive_multiplier]
support_vectors_y = y[has_positive_multiplier]

# Now that we have our alpha values we must compute w and b to find our line

def compute_w(multipliers, X, y):
    return np.sum(multipliers[i] * y[i] * X[i] for i in range(len(y)),)

normw = compute_w(multipliers, X, y);
w = compute_w(sv_multipliers, support_vectors, support_vectors_y)

def compute_b(w, X, y):
    return np.sum([y[i] - np.dot(w, X[i]) for i in range(len(X))])/len(X)

b = compute_b(w,support_vectors,support_vectors_y);

p = ((X.dot(w)) + b >= 0).reshape(m,1);

x_contour = np.linspace(np.min(X[:,0]),np.max(X[:,0]),1000);
y_contour = np.linspace(np.min(X[:,1]),np.max(X[:,1]),1000);

z = np.zeros((len(x_contour),len(y_contour)))

for i in range(len(x_contour)):
    for j in range(len(y_contour)):
            z[i][j] = (x_contour[i] * w[0] + y_contour[j] * w[1]) + b >= 0
            
plt.contour(x_contour,y_contour,z.T,[0]);

plt.show();

y[y==-1] = 0;

print('Final Accuracy: ', np.mean(p == y) * 100)

