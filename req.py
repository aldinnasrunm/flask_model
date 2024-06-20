import requests
import numpy as np

resp = requests.post('http://127.0.0.1:5000/', files={'file': open('acne 2.jpeg', 'rb')})

pred = resp.json()
print(pred)

