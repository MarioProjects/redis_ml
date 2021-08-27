# import the necessary packages

import torch

# -- IMAGE PREPROCESS
"""
pre-processes our input image for classification using the ResNet50 implementation in Keras.. 
When utilizing your own models I would suggest modifying this function to perform 
any required pre-processing, scaling, or normalization.
"""


def tensor_image(image):
    # image is a numpy with shape (1, height, width, channels) -> tensor with (1, channels, height, width)
    return torch.from_numpy(image).permute(0, 3, 1, 2)


# Load the file containing the 1,000 labels for the ImageNet dataset classes
with open('imagenet_classes.txt') as f:
    LABELS = [line.strip() for line in f.readlines()]


# -- AUXILIAR FUNCTIONS
def decode_predictions(out, top=5):
    _, indices = torch.sort(out, descending=True)
    percentages = torch.nn.functional.softmax(out, dim=1)
    return [
        [(LABELS[idx], percentage[idx].item()) for idx in indices[i][:top]]
        for i, percentage in enumerate(percentages)
    ]
