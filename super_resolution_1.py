from super_image import EdsrModel, ImageLoader
from PIL import Image
import argparse
import os
from diffusers.utils import load_image


parser = argparse.ArgumentParser()

parser.add_argument(
    "--workerTaskId",
    type=int,
    default=0,
    help="Worker Task Id"
)

parser.add_argument(
    "--image",
    type=str,
    default="",
    help="Image Path",
)
parser.add_argument(
    "--output",
    type=str,
    default="",
    help="Output Folder",
)

parser.add_argument(
    "--scale",
    type=int,
    default=2,
    help="Output Folder",
)

ap = parser.parse_args()

image_org = load_image(ap.image)

model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=ap.scale)      # scale 2, 3 and 4 models available

preds = model(image_org)

ImageLoader.save_image(preds, '/ap.output/scaled.png') 

