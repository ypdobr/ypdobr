"""Erzeugt data/kickstarter_filtered.csv.gz aus dem Rohabzug von webrobots.io.

Quelle: https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter_2026-09-10T03_20_48_478Z.zip
(Abzug vom 10.09.2026, ein Zip mit 63 CSV-Dateien, etwa 268 MB).

Aufruf: python prepare_data.py PFAD/ZUM/Kickstarter_2026-09-10T03_20_48_478Z.zip
"""
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

COLS = ['id', 'name', 'blurb', 'state', 'goal', 'static_usd_rate', 'launched_at', 'deadline', 'country', 'category']
COUNTRIES = ['US', 'GB', 'CA', 'AU', 'NZ', 'IE']  # englischsprachige Länder
OUT = Path(__file__).parent / 'data' / 'kickstarter_filtered.csv.gz'
LOG = Path(__file__).parent / 'data' / 'prepare_log.json'


def read_all(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.endswith('.csv'))
        frames = [pd.read_csv(z.open(n), usecols=COLS) for n in names]
    return pd.concat(frames, ignore_index=True), len(names)


def main(zip_path):
    raw, n_files = read_all(zip_path)
    log = {'CSV-Dateien': n_files, 'Zeilen roh': len(raw), 'eindeutige ids': raw['id'].nunique()}
    df = raw[raw['state'].isin(['successful', 'failed'])]
    df = df.drop_duplicates('id')  # dieselbe Kampagne erscheint auf mehreren Listenseiten
    log['successful/failed, dedupliziert'] = len(df)
    df = df[df['country'].isin(COUNTRIES)].copy()
    log['englischsprachige Länder'] = len(df)
    cat = df['category'].map(json.loads)
    df['main_cat'] = cat.map(lambda c: c.get('parent_name') or c['name'])
    df['sub_cat'] = cat.map(lambda c: c['slug'])
    df['goal_usd'] = (df['goal'] * df['static_usd_rate']).round(2)
    launched = pd.to_datetime(df['launched_at'], unit='s', utc=True)
    df['launched_year'] = launched.dt.year
    df['duration_days'] = ((df['deadline'] - df['launched_at']) / 86400).round(1)
    df['success'] = (df['state'] == 'successful').astype(int)
    df['name'] = df['name'].fillna('').astype(str).str.strip()
    df['blurb'] = df['blurb'].fillna('').astype(str).str.strip()
    keep = ['id', 'name', 'blurb', 'success', 'goal_usd', 'launched_at', 'launched_year',
            'duration_days', 'country', 'main_cat', 'sub_cat']
    df = df[keep].sort_values(['launched_at', 'id']).reset_index(drop=True)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False, compression={'method': 'gzip', 'compresslevel': 9, 'mtime': 0})
    LOG.write_text(json.dumps({k: int(v) for k, v in log.items()}, ensure_ascii=False, indent=1))
    print(log)
    print(f'geschrieben {OUT}: {len(df):,} Zeilen, {OUT.stat().st_size / 1e6:.1f} MB')
    print(df['launched_year'].value_counts().sort_index().to_string())


if __name__ == '__main__':
    main(sys.argv[1])
