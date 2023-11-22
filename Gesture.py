import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import time 
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import *
from tkinter import ttk, colorchooser
# needs Python Image Library (PIL)
from PIL import Image, ImageDraw
#Import all the enhancement filter from pillow
from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)
from flask import Flask, request
import base64
import numpy as np
import cv2
import time
import json
# ML section

def base64_to_image(base64_uri):
    # Decode the Base64 URI
    data = base64.b64decode(base64_uri.split(',')[1])

    # Convert the binary data to a NumPy array
    image_array = np.frombuffer(data, dtype=np.uint8)

    # Convert the NumPy array to a cv2 image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Return the image
    return image


#tensoflow model loading
model = tf.keras.models.load_model('shapedetector_model_4b.h5')
class_names = ['circle', 'rectangle', 'square', 'triangle']

app = Flask(__name__)


@app.route('/predict', methods = ['POST'])
def predict(): #save the image and push it to model prediction

    data = request.get_json()

    image = base64_to_image(data['imguri'])

    image = Image.fromarray(image)
    png_image = image.convert('RGBA')
    png_image.save('./images/image.png')
    time.sleep(5)
    filename = "./images/image.png"
    img = keras.preprocessing.image.load_img(filename, target_size=(28,28))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])  
    print(class_names[np.argmax(score)])
    print(predictions)
    return json.dumps({"result":class_names[np.argmax(score)]}) 
        

# main function
if __name__ == "__main__":
    app.run(port=5000, debug=True)
