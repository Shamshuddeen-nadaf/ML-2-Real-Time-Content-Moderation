import numpy as np
import onnxruntime as rt
from datasets import load_from_disk
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def main():
    ds = load_from_disk("Data/combined_dataset")
    test = ds["test"]
    print(f"Test samples: {len(test)}")

    tokenizer = AutoTokenizer.from_pretrained("Models/distilbert-toxic")
    session = rt.InferenceSession("Models/distilbert-toxic/model.onnx")

    def tokenize_batch(texts):
        return tokenizer(
            texts, padding="max_length", truncation=True, max_length=128, return_tensors="np"
        )

    batch_size = 1
    all_logits, all_labels = [], []

    texts = test["text"]
    for i in range(0, len(test), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = tokenize_batch(chunk)
        logits = session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })[0]
        all_logits.append(logits)
        all_labels.extend(test["toxic"][i : i + batch_size])
        print(f"  {min(i + batch_size, len(test))}/{len(test)}")

    logits = np.concatenate(all_logits)
    preds = np.argmax(logits, axis=1)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    p, r, f1, _ = precision_recall_fscore_support(all_labels, preds, average="binary")
    auc = roc_auc_score(all_labels, probs[:, 1])

    print(f"\nPrecision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")


if __name__ == "__main__":
    main()
