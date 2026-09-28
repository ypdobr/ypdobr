"""Feinabstimmung von deepset/gbert-base auf dem deutschen Teil von x-stance.

Das Skript erzeugt den Cache, den das Notebook liest. Es trainiert drei Epochen
auf einer nach Frage und Label geschichteten Stichprobe der deutschen
Trainingsdaten und speichert nach jeder Epoche die Softmax-Wahrscheinlichkeiten
für Validierung und die drei deutschen Testsätze in data/cache/. Die Zahl der
Epochen ist vorab auf 3 festgelegt; die Testsätze werden nach jeder Epoche
mitgeschrieben, damit andere Epochen als Sensitivität berichtet werden können.
Alle 200 Schritte und am Ende jeder Epoche wird ein Checkpoint geschrieben, damit
ein abgebrochener Lauf an derselben Stelle weiterläuft.

Aufruf:
    python train_gbert.py --pilot 60          # Durchsatz messen
    python train_gbert.py --n 16925           # Lauf für das Notebook, ergibt 16.911 Paare (Hälfte)
"""
import argparse
import json
import platform
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = "deepset/gbert-base"
SEED = 1977
LR = 2e-5
BATCH = 16
EPOCHS = 3
MAX_LEN = 128
WARMUP = 0.1
WEIGHT_DECAY = 0.01
EVAL_BATCH = 64
LABELS = ["AGAINST", "FAVOR"]  # Index 0 und 1, alphabetisch wie in scikit-learn
TEST_SETS = {"new_comments": "new_comments_defr", "new_questions": "new_questions_defr", "new_topics": "new_topics_defr"}

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / "data" / "xstance-data-v1.0.zip"
CACHE = ROOT / "data" / "cache"
CKPT = Path.home() / ".cache" / "xstance_gbert" / "state.pt"  # außerhalb des Repositorys
CKPT_EVERY = 200


def read_jsonl(name):
    with zipfile.ZipFile(ZIP) as z:
        return pd.read_json(z.open(name), lines=True)


def load_german():
    train = read_jsonl("train.jsonl").query("language == 'de'").reset_index(drop=True)
    dev = read_jsonl("valid.jsonl").query("language == 'de'").reset_index(drop=True)
    test = read_jsonl("test.jsonl").query("language == 'de'").reset_index(drop=True)
    splits = {"dev": dev}
    for short, name in TEST_SETS.items():
        splits[short] = test[test["test_set"] == name].reset_index(drop=True)
    return train, splits


def stratified_subsample(train, n):
    """Zieht je Gruppe (question_id, label) denselben Anteil, mindestens 1 Beispiel."""
    if n <= 0 or n >= len(train):
        return train.copy()
    rng = np.random.default_rng(SEED)
    frac = n / len(train)
    keep = []
    for _, grp in train.groupby(["question_id", "label"], sort=True):
        k = max(1, int(round(frac * len(grp))))
        keep.extend(rng.choice(grp.index.to_numpy(), size=min(k, len(grp)), replace=False).tolist())
    return train.loc[sorted(keep)].reset_index(drop=True)


def encode(tok, df):
    enc = tok(df["question"].tolist(), df["comment"].tolist(), truncation="only_second", max_length=MAX_LEN)
    return [{k: enc[k][i] for k in enc} for i in range(len(df))]


def collate(items, pad_id):
    width = max(len(x["input_ids"]) for x in items)
    batch = {}
    for key in items[0]:
        if key == "labels":
            continue
        fill = pad_id if key == "input_ids" else 0
        batch[key] = torch.tensor([x[key] + [fill] * (width - len(x[key])) for x in items])
    if "labels" in items[0]:
        batch["labels"] = torch.tensor([x["labels"] for x in items])
    return batch


def length_bucketed_batches(n_items, lengths, rng):
    """Mischt, sortiert innerhalb großer Blöcke nach Länge und mischt dann die Batches."""
    order = rng.permutation(n_items)
    block = BATCH * 50
    batches = []
    for start in range(0, n_items, block):
        chunk = sorted(order[start:start + block], key=lambda i: lengths[i])
        batches.extend([chunk[i:i + BATCH] for i in range(0, len(chunk), BATCH)])
    rng.shuffle(batches)
    return batches


@torch.inference_mode()
def predict_proba(model, feats, pad_id):
    model.eval()
    order = np.argsort([len(f["input_ids"]) for f in feats])
    out = np.zeros((len(feats), len(LABELS)), dtype=np.float32)
    for start in range(0, len(order), EVAL_BATCH):
        idx = order[start:start + EVAL_BATCH]
        batch = collate([feats[i] for i in idx], pad_id)
        logits = model(**batch).logits
        out[idx] = torch.softmax(logits, dim=-1).numpy()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="Stichprobengröße, 0 = alle deutschen Trainingsdaten")
    ap.add_argument("--pilot", type=int, default=0, help="nur Durchsatz messen, Anzahl Schritte")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    CACHE.mkdir(parents=True, exist_ok=True)

    train_full, splits = load_german()
    train = stratified_subsample(train_full, args.n)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    pad_id = tok.pad_token_id

    y_train = train["label"].map(LABELS.index).to_numpy()
    feats = encode(tok, train)
    for f, y in zip(feats, y_train):
        f["labels"] = int(y)
    lengths = [len(f["input_ids"]) for f in feats]
    eval_feats = {s: encode(tok, df) for s, df in splits.items()}

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(params, lr=LR)
    rng = np.random.default_rng(SEED)
    steps_per_epoch = int(np.ceil(len(feats) / BATCH))
    total_steps = steps_per_epoch * EPOCHS
    sched = get_linear_schedule_with_warmup(optim, int(WARMUP * total_steps), total_steps)
    model.train()

    if args.pilot:
        batches = length_bucketed_batches(len(feats), lengths, rng)[:args.pilot]
        t0, seen = None, 0
        for step, b in enumerate(batches):
            if step == 5:
                t0, seen = time.time(), 0
            loss = model(**collate([feats[i] for i in b], pad_id)).loss
            loss.backward()
            optim.step(); sched.step(); optim.zero_grad()
            if t0 is not None:
                seen += len(b)
        train_rate = seen / (time.time() - t0)
        t1 = time.time()
        predict_proba(model, eval_feats["dev"][:1024], pad_id)
        eval_rate = 1024 / (time.time() - t1)
        n_eval = sum(len(v) for v in eval_feats.values())
        print(json.dumps({"train_examples_per_s": round(train_rate, 2), "eval_examples_per_s": round(eval_rate, 2),
                          "eval_examples_per_epoch": n_eval, "train_full": len(train_full),
                          "mean_len": float(np.mean(lengths))}))
        return

    (CACHE / "train_subsample_ids.txt").write_text("\n".join(map(str, train["id"].tolist())) + "\n")
    for s, df in splits.items():
        np.save(CACHE / f"ids_{s}.npy", df["id"].to_numpy())
    logfile = CACHE / "train_log.txt"

    def write_log(line):
        with logfile.open("a") as fh:
            fh.write(line + "\n")

    def snapshot(epoch, step, losses, seconds, phase):
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        tmp = CKPT.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                    "torch_rng": torch.get_rng_state(), "log": log, "epoch": epoch, "step": step,
                    "losses": losses, "train_seconds": seconds, "phase": phase}, tmp)
        tmp.replace(CKPT)

    if CKPT.exists():
        state = torch.load(CKPT, weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        torch.set_rng_state(state["torch_rng"])
        log = state["log"]
        log["resumes"] = log.get("resumes", 0) + 1
        epoch0, step0, losses0, seconds0, phase0 = (state["epoch"], state["step"], state["losses"],
                                                   state["train_seconds"], state["phase"])
        write_log(f"RESUMED epoch {epoch0} step {step0} phase {phase0}")
    else:
        log = {"model": MODEL, "seed": SEED, "lr": LR, "batch": BATCH, "epochs": EPOCHS, "max_len": MAX_LEN,
               "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "n_train": len(train), "n_train_full": len(train_full),
               "n_questions_train": int(train["question_id"].nunique()),
               "torch": torch.__version__, "transformers": transformers.__version__,
               "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
               "resumes": 0, "epochs_log": []}
        epoch0, step0, losses0, seconds0, phase0 = 1, 0, [], 0.0, "train"

    for epoch in range(epoch0, EPOCHS + 1):
        first = epoch == epoch0
        losses = list(losses0) if first else []
        seconds_before = seconds0 if first else 0.0
        phase = phase0 if first else "train"
        if phase == "train":
            batches = length_bucketed_batches(len(feats), lengths, np.random.default_rng([SEED, epoch]))
            skip = step0 if first else 0
            t_ep = time.time()
            for step, b in enumerate(batches, 1):
                if step <= skip:
                    continue
                loss = model(**collate([feats[i] for i in b], pad_id)).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step(); optim.zero_grad()
                losses.append(loss.item())
                if step % 50 == 0:
                    el = seconds_before + time.time() - t_ep
                    rate = el / (step - skip) if first and skip else el / step
                    write_log(f"epoch {epoch} step {step}/{len(batches)} loss {np.mean(losses[-50:]):.4f} "
                              f"elapsed {el/60:.1f} min eta_epoch {rate*(len(batches)-step)/60:.1f} min")
                if step % CKPT_EVERY == 0 and step < len(batches):
                    snapshot(epoch, step, losses, seconds_before + time.time() - t_ep, "train")
            t_train = seconds_before + time.time() - t_ep
            snapshot(epoch, len(batches), losses, t_train, "eval")
        else:
            t_train = seconds_before
        ep = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_seconds": round(t_train, 1)}
        t_ev = time.time()
        for s, df in splits.items():
            path = CACHE / f"proba_gbert_ep{epoch}_{s}.npy"
            if phase == "eval" and path.exists():
                proba = np.load(path)  # vor einem Neustart bereits berechnet
            else:
                proba = predict_proba(model, eval_feats[s], pad_id)
                np.save(path, proba)
            y = df["label"].map(LABELS.index).to_numpy()
            pred = proba.argmax(1)
            ep[s] = {"macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
                     "accuracy": round(float(accuracy_score(y, pred)), 4)}
        ep["eval_seconds"] = round(time.time() - t_ev, 1)
        log["epochs_log"].append(ep)
        write_log(f"EPOCH_DONE {json.dumps(ep)}")
        (CACHE / "protokoll_gbert.json").write_text(json.dumps(log, indent=2))
        snapshot(epoch + 1, 0, [], 0.0, "train")
    log["total_seconds"] = round(sum(e["train_seconds"] + e["eval_seconds"] for e in log["epochs_log"]), 1)
    (CACHE / "protokoll_gbert.json").write_text(json.dumps(log, indent=2))
    write_log("TRAINING_FINISHED")
    CKPT.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
