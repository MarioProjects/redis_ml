import os

# -- CONSTANTS

# initialize Redis connection settings
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = 6379
REDIS_DB = 0

# initialize constants used to control image spatial dimensions and data type
# We’ll be passing float32 images to the server with dimensions of 224 x 224 and containing 3 channels.
IMAGE_DTYPE = "float32"
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3

# initialize constants used for server queuing
# Our server can handle a BATCH_SIZE = 32 .
# If you have GPU(s) on your production system, you’ll want to tune your BATCH_SIZE for optimal performance
BATCH_SIZE = 64
IMAGE_QUEUE = "image_queue"

# I’ve found that setting both SERVER_SLEEP and CLIENT_SLEEP to 0.25 seconds
# (the amount of time the server and client will pause before polling Redis again, respectively)
# will work well on most systems. Definitely adjust these constants if you’re building a production system.
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25
