"""Feinabstimmung von roberta-base auf den FOMC-Sätzen (Shah et al. 2023) und Vorhersage für die Statements.

Aufruf im Ordner fomc/:
    python train_roberta.py train     # trainiert, speichert Modell und Test-Wahrscheinlichkeiten
    python train_roberta.py predict   # bewertet alle Sätze aus data/statements.csv

Feste Einstellungen ohne Suche: lr 2e-5, Batch 16, 4 Epochen, max. 128 Tokens, Seed 1977, CPU.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

SEED, LR, BATCH, EPOCHS, MAX_LEN = 1977, 2e-5, 16, 4, 128
MODEL = "roberta-base"
DATA = Path("data")
OUT = Path.home() / ".cache" / "fomc_roberta"
torch.set_num_threads(2)


def predict(model, tok, texts, batch=64):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            probs.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
    return np.concatenate(probs)


def train():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr = pd.read_csv(DATA / "fomc_train.csv")
    te = pd.read_csv(DATA / "fomc_test.csv")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    log = {"model": MODEL, "lr": LR, "batch": BATCH, "epochs": EPOCHS, "max_len": MAX_LEN, "seed": SEED, "loss": []}
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(len(tr))
        losses = []
        for i in range(0, len(tr), BATCH):
            idx = order[i:i + BATCH]
            enc = tok(tr["sentence"].iloc[idx].tolist(), truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt")
            out = model(**enc, labels=torch.tensor(tr["label"].iloc[idx].values))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(out.loss.item())
        log["loss"].append(round(float(np.mean(losses)), 4))
        print(f"epoch {epoch + 1}: loss {log['loss'][-1]}, {time.time() - start:.0f}s", flush=True)
    log["train_seconds"] = round(time.time() - start, 1)
    np.save(DATA / "proba_roberta_test.npy", predict(model, tok, te["sentence"].tolist()))
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    (DATA / "protokoll_roberta.json").write_text(json.dumps(log, indent=1))
    print("done", log, flush=True)


def predict_statements():
    tok = AutoTokenizer.from_pretrained(OUT)
    model = AutoModelForSequenceClassification.from_pretrained(OUT)
    st = pd.read_csv(DATA / "statements.csv")
    np.save(DATA / "proba_roberta_statements.npy", predict(model, tok, st["sentence"].tolist()))
    print("scored", len(st), "statement sentences", flush=True)


if __name__ == "__main__":
    train() if sys.argv[1] == "train" else predict_statements()
