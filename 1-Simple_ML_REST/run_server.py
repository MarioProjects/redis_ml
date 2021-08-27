"""
USAGE

- Start the server -
    python run_server.py

- Submit a request via cURL -
The -X flag and POST value indicates we're performing a POST request.
We supply -F image=@dog.jpg to indicate we're submitting form encoded data.
The image key is then set to the contents of the dog.jpg file.
Supplying the @ prior to dog.jpg implies we would like cURL to load the contents of the image
and pass the data to the request. Finally our endpoint:
    curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'

- Submit a request via Python -
    python simple_request.py
"""

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from flask import Flask, make_response, request, jsonify
import io

# initialize our Flask application and the Keras model
app = Flask(__name__)
model = None

"""
You may be tempted to load your model inside your predict function.
This implies that the model will be loaded each and every time a new request comes in. 
This is incredibly inefficient and can even cause your system to run out of memory.

Your API will run considerably slower (especially if your model is large) — this is due to the 
significant overhead in both I/O and CPU operations used to load your model for each new request.

To see how this can easily overwhelm your server's memory, 
let's suppose we have N incoming requests to our server at the same time. 
This implies there will be N models loaded into memory...again, at the same time. 
If your model is large, such as ResNet, storing N copies of the model in RAM could easily exhaust the system memory.

To this end, try to avoid loading a new model instance for every new incoming request 
unless you have a very specific, justifiable reason for doing so.

Caveat: We are assuming you are using the default Flask server that is single threaded. 
If you deploy to a multi-threaded server you could be in a situation where you are 
still loading multiple models in memory even when using the "more correct" method discussed earlier in this post. 
If you intend on using a dedicated server such as Apache or nginx you should consider making your pipeline
 more scalable, as discussed https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/.
"""


def load_model():
    """
    load the pre-trained Keras model (here we are using a model
    pre-trained on ImageNet and provided by Keras, but you can
    substitute in your own networks just as easily)
    """
    global model
    model = ResNet50(weights="imagenet")


def prepare_image(image, target):
    """
    This function preprocesses an input image prior to passing it through our network for prediction.
    If you are not working with image data you may want to consider changing the name
    to a more generic prepare_datapoint and applying any scaling/normalization you may need
    :param image:
    :param target:
    :return:
    """
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


@app.route("/predict", methods=["POST"])
def predict():
    """
    The actual endpoint of our API that will classify the incoming data from the request
    and return the results to the client
    """

    # initialize the data dictionary that will be returned from the view. Is used to store any data that
    # we want to return to the client. Right now this includes a boolean used to indicate
    # if prediction was successful or not — we'll also use this dictionary to store the results
    # of any predictions we make on the incoming data.
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds, top=5)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                data["predictions"].append({
                    "label": label,
                    "probability": float(prob)
                })

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return make_response(jsonify(data), 200)


if __name__ == "__main__":
    # if this is the main thread of execution first load the model
    # The call to load_model is a blocking operation and
    # prevents the web service from starting until the model is fully loaded.
    print("* Loading Keras model...")
    load_model()

    # then start the server
    print("* Flask started server!")
    app.run()
