from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model('toxicity.h5')
vectorizer = tf.keras.models.load_model('my_vectorizer.keras')

class CommentRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_toxicity(request: CommentRequest):
    vectorized_text = vectorizer([request.text])
    prediction = model.predict(vectorized_text)[0][0]
    return {"is_toxic": prediction > 0.5}
