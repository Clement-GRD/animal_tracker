# from  tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile
import uvicorn
from utils.preprocess_image import read_image_from_buffer, preprocess_image
from utils.predictions import predict
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("models/model/pretrained_fine_300_ratio")

@app.get("/ping")
async def ping():
    return "Server running"

@app.post("/predict")
async def predict_req(file: UploadFile = File(...)):
    image =  read_image_from_buffer(await file.read())
    preprocessed_image =  preprocess_image(image)
    predicted_class, likelihood = predict(preprocessed_image, model)

    return {"predicted_class": predicted_class, "likelihood": likelihood}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)