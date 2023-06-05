import os
import io
import function as func
from fastapi import FastAPI

app = FastAPI()

class Msg(BaseModel):
    msg: str

#@app.get("/")
#async def root():
#    return {"message": "Welcome to the Food Vision API!"}

@app.get('/')
def main():
    return {'message': 'Welcome to Nutriary Model test API!'}
 
## Defining path operation for /name endpoint
#@app.get('/{name}')
#def hello_name(name : str):
#    # Defining a function that takes only string as input and output the
#    # following message.
#    return {'message': f'Welcome to model test FastAPI!, {name}'}


@app.post('/predict')
async def predict_image(file: UploadFile = File(...)):
    # Read and preprocess the image
    image = Image.open(io.BytesIO(await file.read()))
    image = func.preprocess_image(image)

    class_name, probability = predict(image)
    # Return the predicted class
    return {class_name : probability}
    
if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)