#!/bin/python

import Feature
from Iimage import Iimage
from PIL import Image

iimage=Iimage("./faces/face0.jpg")
im=Image.open("./faces/face0.jpg")
pix=im.load()
im.close()

