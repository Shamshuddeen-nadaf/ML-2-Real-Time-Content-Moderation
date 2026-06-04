import gradio as gr
from API.app import softmax, tokenizer, session

def moderate(text):
  inputs = tokenizer(text,return_tensors='np',padding='max_length',max_length=128)
  logits = session.run(None, {"input_ids":inputs["input_ids"],"attention_mask":inputs["attention_mask"]})[0]
  probs = softmax(logits[0]) #type:ignore
  toxic_prob = float(probs[1])
  return "Toxic" if toxic_prob > 0.5 else "Clean", f"{toxic_prob:.4f}"

gr.Interface(
  fn=moderate,
  inputs="text",
  outputs=["text","text"],
  title="Toxic Comment detector"
).launch()

