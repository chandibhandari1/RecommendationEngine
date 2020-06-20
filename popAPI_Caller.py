"""
This is the popularity API caller which provides te userId and no of top songs to recommed
"""
import requests
# if you can give id
# another user = b80344d063b5ccb3212f76538f3d9e43d87dca9e
#json_req = {"userId": '969cc6fb74e076a68e36a04409cb9d3765757508', "K":3}

# since the user id are too long we can provide the user id idex
json_req ={"userId_place": 20, "K":5}
url = ' http://127.0.0.1:5000/popularity_recommender'
req = requests.post(url,json=json_req)
print(req.status_code)
# print the requested result
print(req.json())