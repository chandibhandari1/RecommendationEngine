'''
Created: @Chandi_Bhandari for  Teaching for ML Student
Nearest Neighbor Method: Using Correlation
It has a bug: needed to fix
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


# Select top N movies best on rating of specific user
def topRatedMovie(fix_user, N):
    top_rated_movie = pd.DataFrame.sort_values(
                         data[data.userId==fix_user], ['rating'], ascending=[0])[:N]
    return list(top_rated_movie.title)

# test of the function: top 5 movies for user with userId =63
print('High rated movie for user 63 are \n',list(topRatedMovie(63,5)))


# Create the pivot matrix where one side userId, otherside =itemID and values =rating
userItemRatingMatrix = pd.pivot_table(data, values='rating',
                                      index=['userId'], columns=['itemId'])

# Check out the pivot matrix
print("The Pivot matrix: \n ")
print(userItemRatingMatrix.head())


# Find the similarity between 2 users: a and b
def user_similarity(user1, user2):
    """
    takes users a and b => normalize
    get the common movie
    get the correlation between user a and b using rating values
    """
    user1 = np.array(user1)-np.nanmean(user1)
    user2 = np.array(user2)-np.nanmean(user2)

    # common movies for user1 and user2
    common_movie=[movie for movie in range(len(user1)) if user1[movie]>0 and user2[movie]>0]

    # if they have no common movie: exit otherwise compute the correlation between two vectors of users a and b
    if len(common_movie)==0:
        return 0
    else:
        user1 = np.array(user1[i] for i in common_movie)
        user2 = np.array(user2[i] for i in common_movie)
        return correlation(user1, user2)


# Compute the Nearest Neighbor for a user:  using the correlation from usr_similarity function above
def NNeighborRating(fix_user, K):
    # create the empty matrix with index = userID and column Similarity_score
    similarity_matrix = pd.DataFrame(index=userItemRatingMatrix.index, columns=['Similarity'])
    # find the similarity between fix_user and each user from userItem rating matrix and add value to sim matrix
    for i in userItemRatingMatrix.index:
        similarity_matrix.loc[i] = user_similarity(userItemRatingMatrix.loc[fix_user],
                                                       userItemRatingMatrix.loc[i])
    # Sort the similarity matrix in descending order based on similarity_score
    similarity_matrix = pd.DataFrame.sort_values(similarity_matrix, ['Similarity'], ascending=[0])

    # Chose the top K nearest neighbors
    nearestNeighbor = similarity_matrix[:K]

    # take the neighbor's rating for those movies where fixed_user have not rated (to predict the rating)
    neighborRatingScore = userItemRatingMatrix.loc[nearestNeighbor.index]

    # Now predict the rating for based on other similar users rating for those movies which fix_users have not rated
    predictItemRating = pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating']) # empty matrix
    # for each movie in userItem matrix: start with average rating of users
    for movie in userItemRatingMatrix.columns:
        predictedRating = np.nanmean(userItemRatingMatrix.loc[fix_user])
        # for each neighbor in the neighbor list
        for nbr in neighborRatingScore.index:
            # if nbr has rated movie add that rating adjusted with avg rating of nbr weighted by similarity
            # of the neighbor to the fix_user
            if userItemRatingMatrix.loc[nbr, movie]>0:
                predictedRating += (userItemRatingMatrix.loc[nbr, movie]
                                    -np.nanmean(userItemRatingMatrix.loc[nbr]))\
                                     *nearestNeighbor.loc[nbr,'Similarity']
        # get out of loop and uses nbrs rating to predicted rating matrix
        predictItemRating.loc[movie, 'Rating']= predictedRating
    return predictItemRating

# define function to find the top N recommendation based on what we predicted the rating
def topN_movieRecommendation(fix_user, N):
    # use 10 nearest nbrs to predict the rating
    predictItemRating = NNeighborRating(fix_user, 10)
    # To find the already watched movie = have some rating for that movie
    # find all the not NaN movies (scored by user already)
    seenMovies = list(userItemRatingMatrix.loc[fix_user].loc[userItemRatingMatrix.loc[fix_user]>0].index)
    # drop already seen movies
    predictItemRating = predictItemRating.drop(seenMovies)

    # Get the top recommended movie name
    topRecommendation = pd.DataFrame.sort_values(predictItemRating, ['Rating'], ascending=[0])[:N]

    # get the corresponding movie title
    topRecommendationMovieTitle = (data_m.loc[data_m.itemId.isin(topRecommendation.index)])
    return list(topRecommendationMovieTitle.title)

# Lets test for the fix user
fix_user = 63
# Top rated movie for user: 63
print(topRatedMovie(fix_user,5))

# Not watch but top recommended movie for user 35
print(topN_movieRecommendation(fix_user,10))


