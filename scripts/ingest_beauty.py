"""
One-time ingestion script: load Beauty dataset into PostgreSQL.

Run from project root (conda myenv):
    python scripts/ingest_beauty.py

Connects to the Docker PostgreSQL on localhost:5432.
Timestamps are synthetic (base date + 1 hour per interaction per user).
"""
import os
import pickle
import numpy as np
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import execute_values

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://mlops:mlops@localhost:5432/mlops"
)

DATA_FILE = "data/raw/Beauty_all_multi_word.dat"
BASE_TS = datetime(2020, 1, 1)   # synthetic base timestamp


def load_data():
    with open(DATA_FILE, "rb") as f:
        return pickle.load(f)


def ingest(dat):
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    cur = conn.cursor()

    # ── 1. categories ─────────────────────────────────────────────────────────
    print("Inserting categories...")
    category_rows = [
        (cat_id, name)
        for name, cat_id in dat["category2id"].items()
        if cat_id != 0  # skip [PAD]
    ]
    execute_values(cur,
        'INSERT INTO category (category_id, category) VALUES %s ON CONFLICT DO NOTHING',
        category_rows
    )

    # ── 2. brands ─────────────────────────────────────────────────────────────
    print("Inserting brands...")
    brand_rows = [
        (brand_id, name)
        for name, brand_id in dat["brand2id"].items()
        if brand_id != 0  # skip [PAD]
    ]
    execute_values(cur,
        'INSERT INTO brand (brand_id, brand_name) VALUES %s ON CONFLICT DO NOTHING',
        brand_rows
    )

    # ── 3. items ──────────────────────────────────────────────────────────────
    print("Inserting items...")
    feats = np.array(dat["items_feat"])  # shape (12102, 11)
    # feats layout: [price, cat_id_0..cat_id_8, brand_id]
    item_rows = []
    for item_id in range(1, len(feats)):   # 0 = padding, skip
        row = feats[item_id]
        price = float(row[0]) if row[0] > 0 else None
        # take most specific category IDs (last non-zero = most specific, second-last = parent)
        # aligns with data_generator.py which also uses the last non-zero category
        cats = [int(c) for c in row[1:-1] if c > 0]
        cat1 = cats[-1] if len(cats) > 0 else None
        cat2 = cats[-2] if len(cats) > 1 else None
        brand = int(row[-1]) if row[-1] > 0 else None
        item_rows.append((item_id, cat1, cat2, brand, price))

    execute_values(cur,
        """INSERT INTO item (item_id, category_id1, category_id2, brand_id, price)
           VALUES %s ON CONFLICT DO NOTHING""",
        item_rows
    )

    # ── 4. users ──────────────────────────────────────────────────────────────
    print("Inserting users...")
    user_ids = list(dat["user_seq_token"].keys())
    execute_values(cur,
        'INSERT INTO "user" (user_id) VALUES %s ON CONFLICT DO NOTHING',
        [(uid,) for uid in user_ids]
    )

    # ── 5. interactions ───────────────────────────────────────────────────────
    print("Inserting interactions...")
    interaction_rows = []
    for user_id, seq in dat["user_seq_token"].items():
        for step, (item_id, _features) in enumerate(seq):
            ts = BASE_TS + timedelta(hours=step)
            interaction_rows.append((user_id, item_id, ts))

    execute_values(cur,
        """INSERT INTO interaction (user_id, item_id, timestamp)
           VALUES %s ON CONFLICT DO NOTHING""",
        interaction_rows,
        page_size=1000
    )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done.")
    print(f"  categories : {len(category_rows)}")
    print(f"  brands     : {len(brand_rows)}")
    print(f"  items      : {len(item_rows)}")
    print(f"  users      : {len(user_ids)}")
    print(f"  interactions: {len(interaction_rows)}")


if __name__ == "__main__":
    dat = load_data()
    ingest(dat)
