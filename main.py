from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file 
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims
from tensorflow.nn import softmax
import tensorflow as tf
from numpy import argmax
from numpy import max
from numpy import array
from json import dumps
from uvicorn import run
import numpy as np
from PIL import Image
import os
import io

app = FastAPI()
model_dir = "models/94-stabil-di-test-model-train-1685467718.748879.h5"
model = load_model(model_dir)

class_names = ['bakso', 'gado', 'rendang', 'sate']

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

# app.add_middleware(
#     CORSMiddleware, 
#     allow_origins = origins,
#     allow_credentials = True,
#     allow_methods = methods,
#     allow_headers = headers    
# )

#@app.get("/")
#async def root():
#    return {"message": "Welcome to the Food Vision API!"}

@app.get('/')
def main():
    return {'message': 'Welcome to nutriary model test FastAPI!'}
 
## Defining path operation for /name endpoint
#@app.get('/{name}')
#def hello_name(name : str):
#    # Defining a function that takes only string as input and output the
#    # following message.
#    return {'message': f'Welcome to model test FastAPI!, {name}'}

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match the input size of the model
    image_array = np.array(image)
    image_array = tf.keras.applications.xception.preprocess_input(image_array)
    image_array = np.expand_dims(image, axis=0)
    return image_array

def predict(image):
    # Make the prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    class_name = class_names[predicted_class]
    probability = np.max(prediction) * 100
    
    return class_name, probability

@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
    # Read and preprocess the image
    image = Image.open(io.BytesIO(await file.read()))
    image = preprocess_image(image)

    class_name, probability = predict(image)
    # Return the predicted class
    return {class_name : probability}
    
if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)