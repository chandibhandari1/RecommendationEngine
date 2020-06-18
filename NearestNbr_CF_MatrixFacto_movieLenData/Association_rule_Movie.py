'''
Created: @Chandi_Bhandari for  Teaching for ML Student
Nearest Neighbor Method: Using Association Rule
'''
# importing the general packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# to generate all permutation and combination
import itertools

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

#
allitem=[]
def support(itemset):
    userList = userItemRatingMatrix.index
    nUsers = len(userList)
    ratingMatrix = userItemRatingMatrix
    for item in itemset:
        ratingMatrix = ratingMatrix.loc[ratingMatrix.loc[:,item]>0]
        userList = ratingMatrix.index
    return float(len(userList))/float(nUsers)

# set the min suport: 30% of user watched movie is the min support
minsupport=0.3
for item in list(userItemRatingMatrix.columns):
    itemset = [item]
    if support(itemset)>minsupport:
        allitem.append(item)

# how many left with minsupport
print(len(allitem))

# now for min confidence
minconfidence=0.1
assocRules=[]
i=2
for rule in itertools.permutations(allitem,2):
    #Generates all possible permutations of 2 items from the remaining
    # list of 47 movies
    from_item=[rule[0]]
    to_item=rule
    # each rule is a tuple of 2 items
    confidence=support(to_item)/support(from_item)
    if confidence>minconfidence and support(to_item)>minsupport:
        assocRules.append(rule)

print(assocRules)
