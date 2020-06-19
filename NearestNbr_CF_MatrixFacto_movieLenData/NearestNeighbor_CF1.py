"""
@ Recommendation using  Nearest Neighbor
The Class for Collaborative Filtering: Correlation, Matrix Factorization and Association Rule
"""
import numpy as np
import pandas as pd
# correlation and other packages for N-neighbors
from scipy.spatial.distance import correlation

# create the class which takes data and pivot matrix: userId, itemID, values=rating, K =10 default
class NearestNeighbors:
    def __init__(self, data,data_m,userItemRatingMatrix, K=10):
        self.activeUser = None
        self.N = None
        self.K=K
        self.data_m = data_m
        self.data = data
        self.userItemRatingMatrix= userItemRatingMatrix

    # Find the top N favorite movies of a user
    def favoriteMovies(self, activeUser, N):

        topMovies = pd.DataFrame.sort_values(
            self.data[self.data.userId == activeUser], ['rating'], ascending=[0])[:N]
        return list(topMovies.title)


    # Find the similarity between 2 users: a and b
    def similarity(self, user1, user2):
        """
        takes users a and b => normalize
        get the common movie
        get the correlation between user a and b using rating values
        """
        user1 = np.array(user1) - np.nanmean(user1)
        user2 = np.array(user2) - np.nanmean(user2)
        # common movies for user1 and user2
        commonItemIds = [i for i in range(len(user1)) if user1[i] > 0 and user2[i] > 0]
        # if they have no common movie: exit otherwise compute the correlation between two vectors of users a and b
        if len(commonItemIds) == 0:
            # If there are no movies in common
            return 0
        else:
            user1 = np.array([user1[i] for i in commonItemIds])
            user2 = np.array([user2[i] for i in commonItemIds])
            return correlation(user1, user2)

    # Compute the Nearest Neighbor for a user:  using the correlation from usr_similarity function above
    def nearestNeighbourRatings(self, activeUser, K):
        # create the empty matrix with index = userID and column Similarity_score
        similarityMatrix = pd.DataFrame(index=self.userItemRatingMatrix.index,
                                        columns=['Similarity'])
        # find the similarity between active and each user from userItem rating matrix and add value to sim matrix
        for i in self.userItemRatingMatrix.index:
            similarityMatrix.loc[i] = self.similarity(self.userItemRatingMatrix.loc[activeUser],
                                                 self.userItemRatingMatrix.loc[i])
        # Sort the similarity matrix in descending order based on similarity_score
        similarityMatrix = pd.DataFrame.sort_values(similarityMatrix,
                                                    ['Similarity'], ascending=[0])
        # Chose the top K nearest neighbors
        nearestNeighbours = similarityMatrix[:K]
        # take the neighbor's rating for those movies where fixed_user have not rated (to predict the rating)
        neighbourItemRatings = self.userItemRatingMatrix.loc[nearestNeighbours.index]
        # Now predict the rating for based on other similar users rating for those movies which fix_users have not rated
        predictItemRating = pd.DataFrame(index=self.userItemRatingMatrix.columns, columns=['Rating'])
        # for each movie in userItem matrix: start with average rating of users
        for i in self.userItemRatingMatrix.columns:
            # for each item
            predictedRating = np.nanmean(self.userItemRatingMatrix.loc[activeUser])
            # for each neighbor in the neighbor list
            for j in neighbourItemRatings.index:
                # if nbr has rated movie add that rating adjusted with avg rating of nbr weighted by similarity
                # of the neighbor to the fix_user
                if self.userItemRatingMatrix.loc[j, i] > 0:
                    predictedRating += (self.userItemRatingMatrix.loc[j, i]
                                        - np.nanmean(self.userItemRatingMatrix.loc[j])) * nearestNeighbours.loc[
                                           j, 'Similarity']
            # get out of loop and uses nbrs rating to predicted rating matrix
            predictItemRating.loc[i, 'Rating'] = predictedRating
        return predictItemRating

    # define function to find the top N recommendation based on what we predicted the rating
    def topNRecommendations(self, activeUser, N):
        # use 10 nearest nbrs to predict the rating:default value
        predictItemRating = self.nearestNeighbourRatings(activeUser, self.K)
        # To find the already watched movie = have some rating for that movie
        # find all the not NaN movies (scored by user already)
        moviesAlreadyWatched = list(self.userItemRatingMatrix.loc[activeUser]
                                    .loc[self.userItemRatingMatrix.loc[activeUser] > 0].index)
        # drop already seen movies
        predictItemRating = predictItemRating.drop(moviesAlreadyWatched)
        # Get the top recommended movie name
        topRecommendations = pd.DataFrame.sort_values(predictItemRating,
                                                      ['Rating'], ascending=[0])[:N]
        # get the corresponding movie title
        topRecommendationTitles = (self.data_m.loc[self.data_m.itemId.isin(topRecommendations.index)])
        return list(topRecommendationTitles.title)