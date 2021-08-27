# import the necessary packages
from flask import Flask, request, make_response, jsonify
import torchvision.transforms as transforms
import torchvision.models as models
import torch
from threading import Thread
from PIL import Image
import numpy as np
import base64
import redis
import uuid
import time
import json
import sys
import io
import warnings
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

-- However, there is a subtle problem… --
Depending on how you deploy your deep learning REST API, there is a subtle problem with keeping 
the classify_process function in the same file as the rest of our web API code.

Most web servers, including Apache and nginx, allow for multiple client threads.

If you keep classify_process in the same file as your predict view, then you may load multiple models 
if your server software deems it necessary to create a new thread to serve the incoming client requests 
— for every new thread, a new view will be created, and therefore a new model will be loaded.

The solution is to move classify_process to an entirely separate process and then 
start it along with your Flask web server and Redis server (check Parte3).
"""

# -- CONSTANTS
# initialize constants used to control image spatial dimensions and

# data type
# We’ll be passing float32 images to the server with dimensions of 224 x 224 and containing 3 channels.
IMAGE_DTYPE = "float32"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3

# initialize constants used for server queuing
# Our server can handle a BATCH_SIZE = 32 .
# If you have GPU(s) on your production system, you’ll want to tune your BATCH_SIZE for optimal performance
BATCH_SIZE = 32
IMAGE_QUEUE = "image_queue"
# I’ve found that setting both SERVER_SLEEP and CLIENT_SLEEP to 0.25 seconds
# (the amount of time the server and client will pause before polling Redis again, respectively)
# will work well on most systems. Definitely adjust these constants if you’re building a production system.
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# -- APP
# initialize our Flask application, Redis server, and Keras model
app = Flask(__name__)
# NOTE: Assume that before you run this server script that your Redis server is running. Our Python script
# connect to the Redis store on our localhost on port 6379 (the default host and port values for Redis)
db = redis.StrictRedis(host="localhost", port=6379, db=0)

# -- MODEL
model = None
# Load the file containing the 1,000 labels for the ImageNet dataset classes
with open('imagenet_classes.txt') as f:
    LABELS = [line.strip() for line in f.readlines()]

# -- IMAGE SERIALIZATION
"""
Redis will act as our temporary data store on the server. 
Images will come in to the server via a variety of methods such as cURL, a Python script, or even a mobile app.

Furthermore, images could come in only every once in awhile (a few every hours or days) 
or at a very high rate (multiple per second). We need to put the images somewhere as they queue up 
prior to being processed. Our Redis store will act as the temporary storage.

In order to store our images in Redis, they need to be serialized. Since images are just NumPy arrays, 
we can utilize base64 encoding to serialize the images. Using base64 encoding also has the added 
benefit of allowing us to use JSON to store additional attributes with the image.

Similarly, we need to deserialize our image prior to passing them through our model.
"""


def base64_encode_image(a):
    # base64 encode the input NumPy array
    return base64.b64encode(a).decode("utf-8")


def base64_decode_image(a, dtype, shape):
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")
    # convert the string to a NumPy array using the supplied data
    # type and target shape
    a = np.frombuffer(base64.decodebytes(a), dtype=dtype)
    a = a.reshape(shape)
    # return the decoded image
    return a


# -- IMAGE PREPROCESS
"""
pre-processes our input image for classification using the ResNet50 implementation in Keras.. 
When utilizing your own models I would suggest modifying this function to perform 
any required pre-processing, scaling, or normalization.
"""

PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


def prepare_image(image):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = PREPROCESS(image)
    # Transform tensor back to numpy to store it in redis queue
    # image => tensor (channels, height, width) -> numpy (1, height, width, channels)
    image = np.expand_dims(image.numpy().transpose(1, 2, 0), axis=0)
    # return the image
    return image.astype(IMAGE_DTYPE)


def tensor_image(image):
    # image is a numpy with shape (1, height, width, channels) -> tensor with (1, channels, height, width)
    return torch.from_numpy(image).permute(0, 3, 1, 2)


# -- AUXILIAR FUNCTIONS
def decode_predictions(out, top=5):
    _, indices = torch.sort(out, descending=True)
    percentages = torch.nn.functional.softmax(out, dim=1)
    return [
        [(LABELS[idx], percentage[idx].item()) for idx in indices[i][:top]]
        for i, percentage in enumerate(percentages)
    ]


# -- IMAGE CLASSIFICATION
def classify_process():
    """
    Load the pre-trained Keras model (here we are using a model
    pre-trained on ImageNet and provided by Keras,
    but you can substitute in your own networks just as easily)
    """
    global model
    print("* Loading model...")
    # Loading the model happens only once when this thread is launched —
    # it would be terribly slow if we had to load the model each time we wanted to process an image
    # and furthermore it could lead to a server crash due to memory exhaustion.
    model = models.resnet18(pretrained=True)
    model.eval()
    print("* Model loaded")

    # After loading the model, this thread will continually poll for new images and then classify them
    while True:
        # attempt to grab a batch of images from the redis database, at most, BATCH_SIZE images from our queue
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        # initialize the image IDs and batch of images themselves
        imageIDs = []
        batch = None

        # Loop over the redis queue to generate the batch (if queue not empty)
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"], IMAGE_DTYPE, (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))
            image = tensor_image(image)
            # generate batch stacking images from the queue
            batch = image if batch is None else torch.vstack([batch, image])
            # update the list of image IDs
            imageIDs.append(q["id"])

        # Check if there are any images in our batch
        if len(imageIDs) > 0:
            # If we have a batch of images, we make predictions on the entire batch by passing it through the model
            print("* Batch size: {}".format(batch.shape))
            with torch.no_grad():
                preds = model(batch)
            results = decode_predictions(preds, top=5)

            # Loop over the Redis imageID and their corresponding set of results from our model
            for (imageID, resultSet) in zip(imageIDs, results):
                # initialize the list of output predictions
                output = []
                # Loop over the results and add them to the list of output predictions
                for (label, prob) in resultSet:
                    output.append({"label": label, "probability": float(prob)})

                # Store the output predictions in the database,
                # using the Redis imageID as the key so we can fetch the results
                db.set(imageID, json.dumps(output))

            # remove the set of images from our queue
            db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

        # sleep for a small amount and await the next batch of images to classify.
        time.sleep(SERVER_SLEEP)


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
# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    # load the function used to classify input images in a *separate*
    # thread than the one used for main classification
    print("* Starting model service...")
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()

    # start the web server
    print("* Starting web service...")
    app.run()
