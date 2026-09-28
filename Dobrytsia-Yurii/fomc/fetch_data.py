"""Lädt alle FOMC-Statements 2000 bis 2025 von federalreserve.gov und den Leitzins (Zielwert) von FRED.

Aufruf im Ordner fomc/:  python fetch_data.py
Ergebnis: data/statements.csv (ein Satz pro Zeile) und data/fed_target_rate.csv (Tageswerte).
"""
import io
import re
import time
from html import unescape
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.federalreserve.gov"
DATA = Path("data")
LINKS = [re.compile(r'href="([^"]*?(\d{8})[^"]*?(?:\.htm|/))"[^>]*>\s*Statement\s*<'),  # bis 2020
         re.compile(r'(?s)Statement:</strong>.{0,300}?href="(/newsevents/pressreleases/monetary(\d{8})a\.htm)"')]


def get(url):
    time.sleep(0.2)  # höflich gegenüber dem Server
    return requests.get(url, timeout=30).text


def statement_links():
    pages = [f"{BASE}/monetarypolicy/fomchistorical{y}.htm" for y in range(2000, 2021)]
    pages.append(f"{BASE}/monetarypolicy/fomccalendars.htm")
    links = {}
    for page in pages:
        html = get(page)
        for m in (m for pattern in LINKS for m in pattern.finditer(html)):
            if "2000" <= m.group(2)[:4] <= "2025":
                links.setdefault(m.group(2), set()).add(BASE + m.group(1))
    return dict(sorted(links.items()))


def statement_text(html):
    html = re.sub(r"(?s)<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<(p|br|div|li|h\d)[^>]*>", "\n", html)
    paras = [re.sub(r"\s+", " ", unescape(p)).strip() for p in re.sub(r"<[^>]+>", " ", html).split("\n")]
    keep, started = [], False
    for p in paras:
        if p.startswith(("Voting for", "Voting against", "For media inquiries", "Implementation Note")):
            break  # danach folgen nur Abstimmung und Technik
        if p.startswith(("In a related action", "In taking the discount rate action")):
            continue  # Diskontsatz, keine Aussage zum Kurs
        if re.match(r"(?i)for (immediate )?release", p):
            keep, started = [], True  # der Text beginnt nach der Freigabezeile
            continue
        started = started or ("Committee" in p and len(p.split()) >= 12)
        if started and len(p.split()) >= 8:
            keep.append(p)
    text = " ".join(keep)
    other = ("firmly committed to fulfilling",)  # Strategie-Papier, keine Sitzung
    return text if "federal funds rate" in text and not any(o in text for o in other) else ""


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if len(s.split()) >= 5]


if __name__ == "__main__":
    rows = []
    for date, urls in statement_links().items():
        texts = [statement_text(get(u)) for u in sorted(urls)]
        text = next((t for t in texts if "decided" in t), next((t for t in texts if t), ""))
        for i, s in enumerate(sentences(text)):
            rows.append({"date": pd.Timestamp(date).date(), "sent_id": i, "sentence": s})
    pd.DataFrame(rows).to_csv(DATA / "statements.csv", index=False)
    fred = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
    old = pd.read_csv(io.StringIO(get(fred + "DFEDTAR")), names=["date", "rate"], header=0)
    new = pd.read_csv(io.StringIO(get(fred + "DFEDTARU")), names=["date", "rate"], header=0)
    rate = pd.concat([old, new]).dropna().drop_duplicates("date", keep="last")
    rate[rate["date"] >= "1999-01-01"].to_csv(DATA / "fed_target_rate.csv", index=False)
    print(len(rows), "Sätze aus", len({r["date"] for r in rows}), "Statements;", len(rate), "Zinstage")
