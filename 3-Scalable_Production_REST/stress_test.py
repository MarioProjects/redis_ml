# import the necessary packages
from threading import Thread
import requests
import time

"""
This script will help us to test the server and determine its limitations. 
I always recommend stress testing your deep learning REST API server so that you know if
(and more importantly, when) you need to add additional GPUs, CPUs, or RAM. 
This script kicks off NUM_REQUESTS threads and POSTs to the /predict endpoint.
"""

# initialize the Keras REST API endpoint URL along with the input image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "dog.jpg"

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 100
SLEEP_COUNT = 0.05


def call_predict_endpoint(n):
    # load the input image and construct the payload for the request
    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}

    # submit the request
    r = requests.post(KERAS_REST_API_URL, files=payload)
    response_status = r.status_code

    # ensure the request was successful
    if response_status == 200:
        print("[INFO] thread {} OK".format(n))
    # otherwise, the request failed
    else:
        print("[INFO] thread {} FAILED".format(n))


# loop over the number of threads
threads = []
for i in range(0, NUM_REQUESTS):
    # start a new thread to call the API
    t = Thread(target=call_predict_endpoint, args=(i,))
    threads.append(t)
    t.daemon = True
    t.start()
    time.sleep(SLEEP_COUNT)

# Wait for all threads to finish
for x in threads:
    # join() is a natural blocking call for the join-calling thread to continue after the called thread has terminated
    x.join()

print("Finish!")
