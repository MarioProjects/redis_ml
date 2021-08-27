# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "dog.jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload)
response_status = r.status_code
response_data = r.json()

# ensure the request was successful
if response_status == 200:
    # loop over the predictions and display them
    for (i, result) in enumerate(response_data["predictions"]):
        print(f"{i + 1}. {result['label']}: {result['probability']:.4f}")

# otherwise, the request failed
else:
    print("Request failed")
