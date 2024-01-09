# Импортируем библиотеки

from transformers import pipeline
from pydantic import BaseModel
from fastapi import FastAPI


app = FastAPI()

class Item(BaseModel):
    text: str

classifier = pipeline("sentiment-analysis")

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]