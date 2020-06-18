'''
Created: @Chandi_Bhandari for  Teaching for ML Student
Nearest Neighbor Method: Using Matrix Factorization
'''
# importing the general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# correlation and other packages for N-neighbors
from scipy.spatial.distance import correlation

# import the data
# import the movie rating data
file_path1 = 'C:/Users/pande/Desktop/DataScience/data/u.data'
data_r = pd.read_csv(file_path1, sep='\t', header=None, \
                   names=['userId','itemId','rating','timestamp'])
# looking at the data
print(data_r.head())

# # keep space
print(" ")

# import the movie info data
file_path2 = 'C:/Users/pande/Desktop/DataScience/data/u.item'
data_m = pd.read_csv(file_path2, sep="|", header=None, index_col=False,
                     names=["itemId","title"], usecols=[0,1],encoding = 'latin')
# looking at the data
print(data_m.head())
print(" ")

# Merging the data at the movie id
data = pd.merge(data_r,data_m,left_on='itemId',right_on="itemId")

# sorting the data if we want:first with userId and then itemId
# data=pd.DataFrame.sort_values(data,['userId','itemId'],ascending=[0,1])

# print the data head
print(data.head())
print(" ")

# Counting the unique values for each column
print("The unique values counts of each columns \n",data.nunique())
print(" There are {} unique users.".format(data.userId.nunique()))
print(f"There are {data.itemId.nunique()} unique movies.")

# Create the pivot matrix where one side userId, otherside =itemID and values =rating
userItemRatingMatrix=pd.pivot_table(data, values='rating',
                                    index=['userId'], columns=['itemId'])

# Check out the pivot matrix
print("The Pivot matrix: \n ")
print(userItemRatingMatrix.head())

# define function that does P*Q factorization
def MatrixFactorize(pivotMatrix,K,steps=10, gamma=0.001, lamda=0.02):
    """
    K = no of factors, and uses the Stochastic Gradient descent to find the factor vectors (f1, f2, f3, ..fk)
    lamda = regularization parameter, gamma = learning rate for SGD
    """
    N = len(pivotMatrix.index)
    M = len(pivotMatrix.columns)
    # create matrix with random values: initialization
    P = pd.DataFrame(np.random.rand(N,K), index=pivotMatrix.index)
    Q = pd.DataFrame(np.random.rand(M,K), index=pivotMatrix.columns)

    for step in range(steps):
        for i in pivotMatrix.index:
            for j in pivotMatrix.columns:
                if pivotMatrix.loc[i, j]>0:
                    eij = pivotMatrix.loc[i,j]-np.dot(P.loc[i], Q.loc[j])
                    P.loc[i] = P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])
                    Q.loc[j] = Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])
        e = 0
        for i in pivotMatrix.index:
            for j in pivotMatrix.columns:
                if pivotMatrix.loc[i,j]>0:
                    e = e+pow(pivotMatrix.loc[i,j]-np.dot(P.loc[i],Q.loc[j]), 2) +lamda*(pow(np.linalg.norm(P.loc[i]),2)
                                                                                    +pow(np.linalg.norm(Q.loc[i]),2))
        if e<0.001:
            break
        print(step)
    return P, Q

# Find the Matrix using this function: for only 100 steps
"""
We should do for entire matrix and run for at least 1000 steps but for now run for 100*100 matrix
for 100 steps for simplicity 
"""
(P, Q) = MatrixFactorize(userItemRatingMatrix.iloc[:100,:100], K=2,gamma=0.001,lamda=0.02, steps=100)

# Now test this for specific user
activeUser=63
# Get the rating for all movies
predictItemRating=pd.DataFrame(np.dot(P.loc[activeUser],Q.T),index=Q.index,columns=['Rating'])
# Recommend the movie by sorting the predicted rating
topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:3]
# We found the ratings of all movies by the active user and then sorted them to find the top 3 movies
topRecommendationTitles=data_m.loc[data_m.itemId.isin(topRecommendations.index)]
print("Recommended Movies are: \n",list(topRecommendationTitles.title))