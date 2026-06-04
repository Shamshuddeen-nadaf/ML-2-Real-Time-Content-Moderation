import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_dir", default="Models/distilbert-toxic")
  parser.add_argument("--output", default="Models/distilbert-toxic/model.onnx")
  args = parser.parse_args()

  tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
  model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
  dummy_input = tokenizer("dummy_text",return_tensors="pt",padding="max_length",max_length=128)
  torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        args.output,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
        opset_version=17,
    )

  print(f"ONNX model saved to {args.output}")

if __name__ == '__main__':
  main()
