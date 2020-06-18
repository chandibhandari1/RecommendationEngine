'''
Created: @Chandi_Bhandari for  Teaching for ML Student
Nearest Neighbor Method: Using Correlation
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

# Find the top N favorite movies of a user
def favoriteMovies(activeUser,N):
    topMovies = pd.DataFrame.sort_values(
        data[data.userId==activeUser],['rating'],ascending=[0])[:N]
    return list(topMovies.title)

# Test the result of the function
print(f"Favorite movie for the user are: \n",favoriteMovies(5,3)) # Print the top 3 favorite movies of user 5

# Create the pivot matrix where one side userId, otherside =itemID and values =rating
userItemRatingMatrix=pd.pivot_table(data, values='rating',
                                    index=['userId'], columns=['itemId'])

# Check out the pivot matrix
print("The Pivot matrix: \n ")
print(userItemRatingMatrix.head())

# Find the similarity between 2 users: a and b
def similarity(user1,user2):
    """
    takes users a and b => normalize
    get the common movie
    get the correlation between user a and b using rating values
    """
    user1=np.array(user1)-np.nanmean(user1)
    user2=np.array(user2)-np.nanmean(user2)
    # common movies for user1 and user2
    commonItemIds = [i for i in range(len(user1)) if user1[i] > 0 and user2[i] > 0]
    # if they have no common movie: exit otherwise compute the correlation between two vectors of users a and b
    if len(commonItemIds)==0:
        # If there are no movies in common
        return 0
    else:
        user1=np.array([user1[i] for i in commonItemIds])
        user2=np.array([user2[i] for i in commonItemIds])
        return correlation(user1,user2)

# Compute the Nearest Neighbor for a user:  using the correlation from usr_similarity function above
def nearestNeighbourRatings(activeUser,K):
    # create the empty matrix with index = userID and column Similarity_score
    similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,
                                  columns=['Similarity'])
    # find the similarity between active and each user from userItem rating matrix and add value to sim matrix
    for i in userItemRatingMatrix.index:
        similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],
                                          userItemRatingMatrix.loc[i])
    # Sort the similarity matrix in descending order based on similarity_score
    similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,
                                              ['Similarity'],ascending=[0])
    # Chose the top K nearest neighbors
    nearestNeighbours = similarityMatrix[:K]
    # take the neighbor's rating for those movies where fixed_user have not rated (to predict the rating)
    neighbourItemRatings = userItemRatingMatrix.loc[nearestNeighbours.index]
    # Now predict the rating for based on other similar users rating for those movies which fix_users have not rated
    predictItemRating = pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])
    # for each movie in userItem matrix: start with average rating of users
    for i in userItemRatingMatrix.columns:
        # for each item
        predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])
        # for each neighbor in the neighbor list
        for j in neighbourItemRatings.index:
            # if nbr has rated movie add that rating adjusted with avg rating of nbr weighted by similarity
            # of the neighbor to the fix_user
            if userItemRatingMatrix.loc[j,i]>0:
                predictedRating += (userItemRatingMatrix.loc[j,i]
                                    -np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
        # get out of loop and uses nbrs rating to predicted rating matrix
        predictItemRating.loc[i,'Rating']=predictedRating
    return predictItemRating

# define function to find the top N recommendation based on what we predicted the rating
def topNRecommendations(activeUser,N):
    # use 10 nearest nbrs to predict the rating
    predictItemRating=nearestNeighbourRatings(activeUser,10)
    # To find the already watched movie = have some rating for that movie
    # find all the not NaN movies (scored by user already)
    moviesAlreadyWatched=list(userItemRatingMatrix.loc[activeUser]
                              .loc[userItemRatingMatrix.loc[activeUser]>0].index)
    # drop already seen movies
    predictItemRating=predictItemRating.drop(moviesAlreadyWatched)
    # Get the top recommended movie name
    topRecommendations=pd.DataFrame.sort_values(predictItemRating,
                                                ['Rating'],ascending=[0])[:N]
    # get the corresponding movie title
    topRecommendationTitles=(data_m.loc[data_m.itemId.isin(topRecommendations.index)])
    return list(topRecommendationTitles.title)

# Let's take this for a spin
activeUser=63
print(favoriteMovies(activeUser,5),"\n")
print(topNRecommendations(activeUser,10))
