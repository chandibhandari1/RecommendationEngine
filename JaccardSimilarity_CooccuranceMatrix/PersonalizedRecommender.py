"""
Teaching Recommendation for ML Students:
Collaborative Filtering using: Jaccard Similarity and Cooccurance
This class has Similarity based Personalized recommender: when we need to recommend the item similarity
user-item similarity.
"""
import numpy as np
import pandas as pd

# Define a class for popularity model
class personalizedRecommendation():
    def __init__(self):
        self.data = None
        self.userId = None
        self.songId = None
        self.cooccurence_matrix = None
        self.personalizedSongs = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.itemSimilarityRecommendation = None

    # define the functions which returns the number of unique users, unique songs and total data
    def uniqueValues(self, df):
        user_count = df.userId.nunique(dropna=True)
        songs_count = df.songId.nunique(dropna=True)
        total_data = len(df)
        return f"Total data: {total_data}, \nunique users: {user_count} and \nunique songs: {songs_count}"

    # create the instances as the given infos
    def create_table(self, data, userId, songId):
        self.data = data
        self.userId = userId
        self.songId = songId

    # get all the songs in the list format
    def getAllSongs(self):
        allSongs = list(self.data[self.songId].unique())
        return allSongs

    # get the unique songs for the given user  (user- item)
    def get_user_item(self, user):
        user_data = self.data[self.data[self.userId]==user]
        user_item = list(user_data[self.songId].unique())
        return user_item

    # Get the unique users for the given song
    def get_item_user(self,song):
        song_data = self.data[self.data[self.songId]==song]
        item_user = list(song_data[self.userId].unique())
        return item_user

    # function to create the cooccurence matrix
    def construct_cooccurenceMatrix(self, user_songs, all_songs):
        # user in user_songs data
        user_songs_users =[]
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_user(user_songs[i]))
            # create the empty matrix
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))),float)
        # calculate the similarity  between user songs and all unique songs in the data
        for i in range(0, len(all_songs)):
            # calculate the unique users of song i
            skills_i_data = self.data[self.data[self.songId] == all_songs[i]]
            users_i = set(skills_i_data[self.userId].unique())
            for j in range(0,len(user_songs)):
                # Get the unique users of songs
                users_j = user_songs_users[j]
                # calculate the intersection of users of song i and song j
                users_intersection = users_i.intersection(users_j)
                # calculate the cooccurence matrix[i,j] as Jaccard Index (Jaccard Similarity)
                if len(users_intersection) == 0:
                    cooccurence_matrix[j, i] = 0
                else:
                    # calculate the union of users of song i and song j
                    users_union = users_i.union(users_j)
                    cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
        return cooccurence_matrix

    # Top recommendation: similarity recommendation using cooccurence matrix
    def topRecommendation(self,user, cooccurence_matrix, all_songs, user_songs):
        non_zero_value = np.count_nonzero(cooccurence_matrix)
        print("Non-zero values in cooccurence matrix: %d"%non_zero_value)
        # calculate a weighted average of the score in cooccurence matrix for all user songs
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
        # sort the indices of user_sim_score based upon their values and also maintain the corresponding score
        sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)
        # create the data frame with columns and fill the df
        df = pd.DataFrame(columns=['userId', 'song', 'score', 'Rank'])
        Rank = 1
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and Rank <=10:
                df.loc[len(df)] = [user, all_songs[sort_index[i][1]],sort_index[i][0] ,Rank]
                Rank +=1
        # when cooccurence matrix has zero values
        if df.shape[0] == 0:
            print('Current user has no songs for training item similarity based Recommendation Model')
            return -1
        else:
            return df

    # define function to recommendation to recommend the similarity based recommendation
    def recommend(self, user):
        """
        user_songs: get all unique song for user
        all_songs: get all unique song in the data
        """
        user_songs = self.get_user_item(user)
        all_songs = self.getAllSongs()
        cooccurence_matrix = self.cooccurence_matrix(user_songs, all_songs)
        recommended_song_df = self.topRecommendation(user, cooccurence_matrix, all_songs, user_songs)
        return recommended_song_df

    # function to recommend the similar song to the given song
    def SimilarSongRecommendation(self, song_list):
        user_songs = song_list
        all_songs = self.getAllSongs()
        print("There are %s unique songs in the data provided" %len(all_songs))
        # create the co-occurance matrix size: len(user_song)*len(allSongs)
        cooccurence_matrix = self.construct_cooccurenceMatrix(user_songs,all_songs)
        # make Top recommendation
        user = ""
        recommended_song_df = self.topRecommendation(user, cooccurence_matrix, all_songs, user_songs)
        return recommended_song_df