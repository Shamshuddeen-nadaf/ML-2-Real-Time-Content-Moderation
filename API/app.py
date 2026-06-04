from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort 
from transformers import AutoTokenizer
import numpy as np

app = FastAPI(title="Real time content moderation")

tokenizer = AutoTokenizer.from_pretrained("Models/distilbert-toxic")
session = ort.InferenceSession("Models/distilbert-toxic/model.onnx")

class ModerateRequest(BaseModel):
  text:str

class ModerateResponse(BaseModel):
  toxic:bool
  confidence:float
  probability:float

def softmax(x):
  e_x = np.exp(x-np.max(x))
  return e_x/e_x.sum()


@app.post("/moderate",response_model=ModerateResponse)
def moderate(req:ModerateRequest):
  inputs = tokenizer(req.text,return_tensors='np',padding='max_length',max_length=128,truncation=True)
  logits= session.run(None, {"input_ids":inputs["input_ids"],"attention_mask":inputs["attention_mask"]})[0]
  probs = softmax(logits[0]) #type:ignore
  toxic_prob = float(probs[1])
  return ModerateResponse(
    toxic = toxic_prob > 0.5,
    confidence=float(np.max(probs)),
    probability = toxic_prob,
  )
