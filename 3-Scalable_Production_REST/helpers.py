# import the necessary packages
import numpy as np
import base64
import sys

import torchvision.transforms as transforms

from settings import *

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
