# import the necessary packages
from flask import Flask, request, make_response, jsonify
from PIL import Image
import redis
import uuid
import time
import json
import io
import warnings

from helpers import base64_encode_image, prepare_image
from settings import *

warnings.filterwarnings('ignore')

"""
-- Considerations when scaling your deep learning REST API --
If you anticipate heavy load for extended periods of time on your deep learning REST API 
you may want to consider a load balancing algorithm such as round-robin scheduling 
to help evenly distribute requests across multiple GPU machines and Redis servers.

Keep in mind that Redis is an in-memory data store so 
we can only store as many images in the queue we have available memory.

A single 224 x 224 x 3 image with a float32 data type will consume 60,2112 bytes of memory.
Assuming a server with a modest 16GB of RAM, this implies that we can hold approximately 26,500 images in our queue, 
but at that point we likely would want to add more GPU servers to burn through the queue faster.
"""

# -- APP
# initialize our Flask application, Redis server, and Keras model
app = Flask(__name__)
# NOTE: Assume that before you run this server script that your Redis server is running. Our Python script
# connect to the Redis store on our localhost on port 6379 (the default host and port values for Redis)
db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


@app.route("/")
def homepage():
    return "Welcome to the PyImageSearch Keras REST API!"


# -- PREDICT ENDPOINT
@app.route("/predict", methods=["POST"])
def predict():
    data = {}  # Generate result dictionary
    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format and prepare it for classification
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image)

            # ensure our NumPy array is C-contiguous as well, otherwise we won't be able to serialize it
            # You can see more about C-contiguous ordering -> https://kutt.it/jn0PwD
            image = image.copy(order="C")

            # generate an unique ID for the classification object we send to the queue
            k = str(uuid.uuid4())
            # Add classification ID + the base64 encoding of the image to the queue
            d = {"id": k, "image": base64_encode_image(image)}
            db.rpush(IMAGE_QUEUE, json.dumps(d))

            # keep looping until our model server returns the output predictions
            while True:
                # attempt to grab the output predictions
                output = db.get(k)

                # check to see if our model has classified the input image
                if output is not None:
                    # add the output predictions to our data dictionary so we can return it to the client
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    # delete the result from the database, since we have pulled the results from the database
                    # and no longer need to store them in the database, and break from the polling loop
                    db.delete(k)
                    break

                # if image not classified sleep a small amount to give the model a chance to classify the image
                # Note: For this example script, I didn't bother adding timeout logic in the above loop
                # which would ideally add a success value of False to the data.
                # I’ll leave that up to you to handle and implement.
                time.sleep(CLIENT_SLEEP)

        if not bool(data):  # Check if request was not POST or not files named 'image'
            return make_response(jsonify({}), 400)

        # return the data dictionary as a JSON response
        return make_response(jsonify(data), 200)


# -- START THE SERVER
if __name__ == "__main__":
    # start the web server
    print("* Starting web service...")
    app.run()
