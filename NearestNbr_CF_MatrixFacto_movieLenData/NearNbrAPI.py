"""
Recommendation API: by Chandi Bhandari
note: Run this API first and then run request.py file,but keep in mind NearestNeighbor_CF should be on
same folder
"""
import numpy as np
import pandas as pd
import requests
import sys

# append the path that for manual foler where install the python packages
sys.path.append('C:\\User\pande\Documents\Python_packages')

#import joblib
from joblib import dump, load
from flask import Flask, request, jsonify, make_response
from flask import render_template

from io import StringIO

"""Here I have loaded the data, merged two tables and """

# data reading:
# import the movie rating data
file_path1 = 'C:/Users/pande/Desktop/DataScience/ByteSizedRecEng_udemy/MovieLens_data/ml-100k/ml-100k/u.data'
data_r = pd.read_csv(file_path1, sep='\t', header=None, \
                   names=['userId','itemId','rating','timestamp'])

# import the movie info data
file_path2 = 'C:/Users/pande/Desktop/DataScience/ByteSizedRecEng_udemy/MovieLens_data/ml-100k/ml-100k/u.item'
data_m = pd.read_csv(file_path2, sep="|", header=None, index_col=False,
                     names=["itemId","title"], usecols=[0,1],encoding = 'latin')

############################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
I have done the following process in Class object: NearestNeighbors in module Nearesteighbor_CF
if I want to pass the data and userItemRatingMatrix, I should instantiate as:
import NearestNeighbor_CF as NearestNeighborCF
NearestNbrCF_model = NearestNeighborCF.NearestNeighbors(data = data,userItemRatingMatrix=userItemRatingMatrix)
and change __init__ (self, data, userItemRatingMatrix, K=10) in the module
'''

# # Merging the data at the movie id
# data = pd.merge(data_r,data_m,left_on='itemId',right_on="itemId")
# # print the data head
# print(data.head())
# print(" ")
#
# # Create the pivot matrix where one side userId, otherside =itemID and values =rating
# userItemRatingMatrix=pd.pivot_table(data, values='rating',
#                                     index=['userId'], columns=['itemId'])
# # Check out the pivot matrix
# print("The Pivot matrix: \n ")
# print(userItemRatingMatrix.head())

################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Import my class object that I wrote on NearestNeighbor_CF.py file
import NearestNeighbor_CF as NearestNeighborCF
NearestNbrCF_model = NearestNeighborCF.NearestNeighbors(data_r = data_r,data_m=data_m)


# Favorite movie: highly rated movie for specific user:
"""
Since our function inside the object: NearestNbrCF_model
 """

app = Flask(__name__)

@app.route('/note', methods=['GET', 'POST'])
def index():
    return "Hello this RecEngine"

"""
To print this note type http://127.0.0.1:5000/note in the browser
You will see: "Hello this RecEngine" in your browser
"""

@app.route('/favorite_movie', methods=['POST'])
def favorite_movie():
    print("Recommendation Started ")
    favorite_movie=[]
    req=request.get_json()
    activeUser = req['userId']
    N = req['noMovieRequested']
    print('Favorite movie for:%s'%activeUser)
    topRatedMovie = NearestNbrCF_model.favoriteMovies(activeUser,N)
    favorite_movie.append(topRatedMovie)
    print('Favorite movie', topRatedMovie)
    return jsonify({"Top Movie":favorite_movie, "\\nOnly 3 Movie":favorite_movie[0][0:3]})


@app.route('/recommended_movie', methods=['POST'])
def recommended_movie():
    print("Recommendation Started ")
    Recommended_movie = []
    req = request.get_json()
    activeUser = req['userId']
    N = req['noMovieRequested']
    #K = req['K']
    print('Recommended movie for:%s' % activeUser)
    predictedMovie = NearestNbrCF_model.topNRecommendations(activeUser, N)
    Recommended_movie.append(predictedMovie)
    return jsonify({"Recommended Movie": Recommended_movie, "\nOnly 3 Movie": Recommended_movie[0][0:3]})

if __name__ == '__main__':
    app.run(debug=True, threaded=False)

