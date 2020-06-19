import requests
# give the userID, top movie to display, Top Recomended movie count to display
json_req ={"userId":63, "noMovieRequested":5, "K":10}

# For Favorite Movie
# # # host and port number and API function to call
# url = ' http://127.0.0.1:5000/favorite_movie'
# # create the post request
# req = requests.post(url,json=json_req)
# print(req.status_code)
# # print the requested result
# print(req.json())

# For Recommended movie
# host and port number and API function to call
url = ' http://127.0.0.1:5000/recommended_movie'
# create the post request
req = requests.post(url,json=json_req)
print(req.status_code)
# print the requested result
print(req.json())

