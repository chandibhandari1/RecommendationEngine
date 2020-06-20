"""
Teaching Recommendation for ML Students:
This class has popularity based recommender: when we need to recommend the trending
songs, movie or text or speech, this model is useful. However it doesn't contain more than
sorting and recommending the top values
"""
import numpy as np
import pandas as pd

# Define a class for popularity model
class popularityRecommendation():
    def __init__(self):
        self.data = None
        self.userId = None
        self.songId = None
        self.popularSongs=None

    # function to display most popular song
    def mostListenedSongs(self, df):
        Songs_listen_count=df.groupby(['song']).agg({'count_listen':'sum'}).reset_index()
        sum_listen = Songs_listen_count['count_listen'].sum()
        Songs_listen_count['percentage'] = Songs_listen_count['count_listen'].div(sum_listen) * 100
        Songs_listen_count = Songs_listen_count.sort_values(['count_listen'], ascending=[0])
        return Songs_listen_count

    # define the functions which returns the number of unique users, unique songs and total data
    def uniqueValues(self, df):
        user_count = df.userId.nunique(dropna=True)
        songs_count = df.songId.nunique(dropna=True)
        total_data = len(df)
        return f"Total data: {total_data}, \nunique users: {user_count} and \nunique songs: {songs_count}"

    # define a function which takes data, userId, itemId and  K = no of top songs we want
    def table_sort(self, data, userId, songId, K=10):
        """
        this function takes take, counts no of user for each movie
        sort values in descending order then recommend top K values
        """
        self.data = data
        self.userId = userId
        self.songId = songId
        self.K = K
        Songs_user_count = data.groupby([self.songId]).agg({self.userId:'count'}).reset_index()
        Songs_user_count.rename(columns = {'userId':'ListenerCount'},inplace=True)
        sorted_data = Songs_user_count.sort_values(['ListenerCount',self.songId], ascending=[0,1])
        sorted_data['Order'] = sorted_data['ListenerCount'].rank(method='first',ascending=False)
        self.popularSongs = sorted_data.head(self.K)

    def recommendation(self,userId):
        recommended_song = self.popularSongs
        recommended_song['userId'] = userId
        # bring userId column in the first
        column_switch = recommended_song.columns.tolist()
        # bring last [-1] to front and all other after that
        column_switch = column_switch[-1:]+column_switch[:-1]
        user_recommendation = recommended_song[column_switch]
        return user_recommendation
