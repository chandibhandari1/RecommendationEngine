"""
@ Teaching ML students to create API for our popularity model
by: Chandi Bhandari
We have used the data from:songCount: https://static.turi.com/datasets/millionsong/10000.txt
                           SongsInfo: https://static.turi.com/datasets/millionsong/song_data.csv
"""
import sys
sys.path.append('C:\\User\pande\Documents\Python_packages')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from flask import render_template, make_response
# import the module that we created for popularity recommendation
import PopularityRecommender as popRecEng
Popularity_model = popRecEng.popularityRecommendation()

# loading data
# with open('data/Song_listener.txt') as f:
#     Listener_count = f.read()
# f.closed
# Listener_count = pd.read_table(Listener_count, header =None)
#Listener_count.columns =['userId', 'songId', 'count_listen']

# Loading Lister count data
Lister_count_dataPath ='data/Song_listener.txt'
column_names =['userId', 'songId', 'count_listen']
Listener_count = pd.read_csv(Lister_count_dataPath, sep='\t', names =column_names)

# Loading the songs information
song_data_path = 'data/song_data.csv'
song_data = pd.read_csv(song_data_path)
song_data.rename(columns ={'song_id':'songId'}, inplace=True)
song_data.drop_duplicates(['songId'])


# Merging the data set with songId
full_data = pd.merge(Listener_count, song_data, on='songId', how='left')
# merge the title and artist
full_data['song'] = full_data['title'].map(str)+"-"+full_data['artist_name']
full_data = full_data[['userId', 'songId','song' , 'count_listen']]
print(" ")
print(full_data.head())

# separate the data for Train and Test_purpose
data, TestData = train_test_split(full_data,test_size=0.20, random_state=1)

# Get the unique values
uniqueValues = Popularity_model.uniqueValues(full_data)
print(uniqueValues)

# call the function to show most listend songs
mostListendSong = Popularity_model.mostListenedSongs(data)
print(mostListendSong.head())

# import our module
import PopularityRecommender as recEng
Popularity_model=recEng.popularityRecommendation()
# to get the id: and to get any nuber to show provide K otherwise it take K=10
#Popularity_model.table_sort(data, "userId", 'songId')
Popularity_model.table_sort(data, "userId", 'song')

app = Flask(__name__)

@app.route('/note', methods=['GET', 'POST'])
def index():
    return "Hello this is popularity Based Recommender"


@app.route('/popularity_recommender', methods=['POST'])
def popularity_recommender():
    print("Recommendation started")
    most_popular_songs =[]
    req = request.get_json()
    userId = req['userId']
    K = req['K']
    print(f'Recommended  Top {K} popular songs for {userId} are: \n')
    recommended_songs = Popularity_model.recommendation(userId)
    # for the song id
    #most_popular_songs.append(recommended_songs['songId'].tolist())
    most_popular_songs.append(recommended_songs['song'].tolist())
    #pop_song_id = most_popular_songs[0][0:K]
    print(most_popular_songs[0][0:K])
    # popularSongs = list(data.loc[data.songId.isin(pop_song_id)].title)
    # print('Top 5 popular songs are: \n', popularSongs)
    return jsonify({"Songs":most_popular_songs[0][0:K]})

if __name__ =='__main__':
    app.run(debug=True, threaded=False)





