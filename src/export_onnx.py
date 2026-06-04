import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort

def export_to_onnx(model_name, model_dir, output_path, quantize=False):
  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  model = AutoModelForSequenceClassification.from_pretrained(model_dir)
  dummy_input = tokenizer("dummy_text",return_tensors="pt",padding="max_length",max_length=128)
  torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
        opset_version=17,
    )

  if quantize:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quant_path = output_path.replace(".onnx", "_int8.onnx")
    quantize_dynamic(output_path, quant_path, weight_type=QuantType.QUInt8)
    output_path = quant_path

  return output_path

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_dir", default="Models/distilbert-toxic")
  parser.add_argument("--output", default="Models/distilbert-toxic/model.onnx")
  parser.add_argument("--quantize", action="store_true")
  args = parser.parse_args()
  path = export_to_onnx(
    model_name = "distilbert-base-uncased",
    model_dir=args.model_dir,
    output_path=args.output,
    quantize=args.quantize,
  )
  print(f"ONNX model saved to {path = }")

if __name__ == '__main__':
  main()