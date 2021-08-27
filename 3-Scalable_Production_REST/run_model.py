import time
import torch
import torchvision.models as models
import json
import redis

import warnings
warnings.filterwarnings('ignore')

from helpers_model import *
from helpers import *
from settings import *

"""
-- NOTE --
keep in mind that your machine will still be limited by I/O. 
It may be beneficial to instead utilize multiple machines, 
each with 1-4 GPUs than trying to scale to 8 or 16 GPUs on a single machine.

-- Recommendations for deploying your own deep learning models to production --
One of the best pieces of advice I can give is to keep your data, in particular your Redis server, close to the GPU.

You may be tempted to spin up a giant Redis server with hundreds of gigabytes of RAM 
to handle multiple image queues and serve multiple GPU machines.

The problem here will be I/O latency and network overhead.

Assuming 224 x 224 x 3 images represented as float32 array, a batch size of 32 images will be ~19MB of data. 
This implies that for each batch request from a model server, 
Redis will need to pull out 19MB of data and send it to the server.

On fast switches this isn’t a big deal, but you should consider running both your model server and Redis 
on the same server to keep your data close to the GPU.
"""

# NOTE: Assume that before you run this server script that your Redis server is running. Our Python script
# connect to the Redis store on our localhost on port 6379 (the default host and port values for Redis)
db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


# -- IMAGE CLASSIFICATION
def classify_process():
    """
    Load the pre-trained Keras model (here we are using a model
    pre-trained on ImageNet and provided by Keras,
    but you can substitute in your own networks just as easily)
    """
    print("* Loading model...")
    # Loading the model happens only once when this thread is launched —
    # it would be terribly slow if we had to load the model each time we wanted to process an image
    # and furthermore it could lead to a server crash due to memory exhaustion.
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True).to(DEVICE)
    model.eval()
    print(f"* Model loaded on {DEVICE}")

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
                preds = model(batch.to(DEVICE))
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


# if this is the main thread of execution start the model server process
if __name__ == "__main__":
    classify_process()
