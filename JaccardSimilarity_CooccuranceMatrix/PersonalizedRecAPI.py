"""
@ Teaching ML students to create API for our (Similarity)Personalized model
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
import PersonalizedRecommender as personalizedRecEng
personalized_model = personalizedRecEng.personalizedRecommendation()

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
uniqueValues = personalized_model.uniqueValues(full_data)
print(uniqueValues)

# since the id are too long we can provide the index
users = data.userId.unique()
Song_id = data.songId.unique()
Songs = data.songId.unique()

# create the table
personalized_model.create_table(data, 'userId', 'song')

app = Flask(__name__)

@app.route('/note', methods=['GET', 'POST'])
def index():
    return "Hello this is popularity Based Recommender"


@app.route('/personalized_recommender', methods=['POST'])
def personalized_recommender():
    print("Recommendation started")
    personalized_songs =[]
    req = request.get_json()
    # if userId itself is given
    userId = req['userId']
    # if userId location index is given
    # userId = users[req["userId_place"]]
    K = req['K']
    print(f'Recommended  Top {K} personalized songs for {userId} are: \n')
    recommended_songs = personalized_model.recommend(userId)
    # for the song id
    print(recommended_songs.columns)
    personalized_songs.append(recommended_songs['song'].tolist())
    print(f'personalized songs for user: {userId}')
    print(personalized_songs[0][0:K])
    return jsonify({"Songs":personalized_songs[0][0:K]})

@app.route('/itemSimilarity_recommender', methods=['POST'])
def itemSimilarity_recommender():
    print("Recommendation started")
    similar_songs =[]
    req = request.get_json()
    # if song itself is given
    song = req['songName']
    song = [song]
    # if song location index is given SimilarSongRecommendation
    #song = Songs[req["songName_place"]]
    K = req['K']
    recommended_songs = personalized_model.SimilarSongRecommendation(song)
    # for the song id
    #most_popular_songs.append(recommended_songs['songId'].tolist())
    similar_songs.append(recommended_songs['song'].tolist())
    print('Recommended similar song for %s'%song)
    print(similar_songs)
    print(f'Recommended  Top {K} similar songs for {song} are: \n')
    print(similar_songs[0][0:K])
    return jsonify({"Songs":most_popular_songs[0][0:K]})

if __name__ =='__main__':
    app.run(debug=True, threaded=False)



