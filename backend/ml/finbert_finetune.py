"""Fine-tune FinBERT on financial news sentiment and compare it with simple baselines.

The original FinBERT was trained on a different dataset, so this set is new to it. The
official validation file is held back as the test set and used once, after training.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from filinglens import config  # noqa: E402

BASE_MODEL = "ProsusAI/finbert"
DATASET = "zeroshot/twitter-financial-news-sentiment"
OUTPUT_DIR = BACKEND / "models" / "finbert-finance"
CHECKPOINTS = config.DATA_DIR / "ml" / "finbert-checkpoints"
RESULTS_PATH = BACKEND / "eval" / "results" / "finbert.json"

SEED = 0
MAX_LENGTH = 96
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
BOOTSTRAP_SAMPLES = 2000

# The dataset uses 0=negative, 1=positive, 2=neutral. FinBERT uses 0=positive, 1=negative, 2=neutral.
TO_FINBERT = {0: 1, 1: 0, 2: 2}


def score(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(SEED)
    samples = [
        f1_score(y_true[pick], y_pred[pick], average="macro")
        for pick in (rng.integers(0, len(y_true), len(y_true)) for _ in range(BOOTSTRAP_SAMPLES))
    ]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_f1_ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
    }


def predict(model, tokenizer, texts: list[str], batch_size: int = 16) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
            ).to(device)
            predictions.append(model(**batch).logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(predictions)


def main() -> None:
    raw = load_dataset(DATASET)
    full = raw["train"].map(lambda row: {"label": TO_FINBERT[row["label"]]}).shuffle(seed=SEED)
    test = raw["validation"].map(lambda row: {"label": TO_FINBERT[row["label"]]})
    split = full.train_test_split(test_size=0.1, seed=SEED)
    train, dev = split["train"], split["test"]
    y_test = test["label"]

    majority = int(np.bincount(train["label"]).argmax())
    results = {
        "sizes": {"train": len(train), "dev": len(dev), "test": len(test)},
        "test": {"majority_class": score(y_test, np.full(len(y_test), majority))},
    }

    tfidf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    tfidf.fit(train["text"], train["label"])
    results["test"]["tfidf_logistic"] = score(y_test, tfidf.predict(test["text"]))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    original = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL).to(device)
    results["test"]["original_finbert"] = score(y_test, predict(original, tokenizer, test["text"]))
    print(json.dumps(results, indent=2), flush=True)
    del original

    def encode(rows):
        return tokenizer(rows["text"], truncation=True, max_length=MAX_LENGTH)

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(CHECKPOINTS),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=16,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            seed=SEED,
            report_to=[],
            logging_steps=100,
        ),
        train_dataset=train.map(encode, batched=True),
        eval_dataset=dev.map(encode, batched=True),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")},
    )
    trainer.train()

    predictions = predict(model, tokenizer, test["text"])
    results["test"]["finetuned_finbert"] = score(y_test, predictions)

    tfidf_correct = np.asarray(tfidf.predict(test["text"])) == np.asarray(y_test)
    tuned_correct = predictions == np.asarray(y_test)
    results["finetuned_minus_tfidf_accuracy"] = float(tuned_correct.mean() - tfidf_correct.mean())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
