import argparse
import numpy as np
from datasets import load_from_disk
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--data_dir", default="Data/combined_dataset")
    parser.add_argument("--output_dir", default="models/distilbert-toxic")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    dataset = load_from_disk(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=128
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("toxic", "labels")
    tokenized.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_dir=f"{args.output_dir}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"], # type: ignore
        eval_dataset=tokenized["test"], # type: ignore
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()