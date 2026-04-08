"""
Migration: fix item category assignments in DB.

The original ingest_beauty.py used cats[0] (least specific, always "Beauty").
This script updates category_id1/category_id2 to use the most specific category
(last non-zero), aligning with data_generator.py's logic.

Usage:
    python -m scripts.migrate_item_categories
"""

import os
import pickle

import numpy as np
import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]
DATA_FILE = os.getenv("DATA_FILE", "/app/data/raw/Beauty_all_multi_word.dat")


def main():
    with open(DATA_FILE, "rb") as f:
        dat = pickle.load(f)

    feats = np.array(dat["items_feat"])

    updates = []
    for item_id in range(1, len(feats)):
        row = feats[item_id]
        cats = [int(c) for c in row[1:-1] if c > 0]
        cat1 = cats[-1] if len(cats) > 0 else None
        cat2 = cats[-2] if len(cats) > 1 else None
        updates.append((cat1, cat2, item_id))

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE item SET category_id1 = %s, category_id2 = %s WHERE item_id = %s",
                updates,
            )
        conn.commit()
        print(f"Done. Updated {len(updates)} items.")

        # Verify
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.category, COUNT(*) as cnt
                FROM item i
                JOIN category c ON i.category_id1 = c.category_id
                GROUP BY c.category
                ORDER BY cnt DESC
                LIMIT 10
            """)
            print("\nTop 10 categories after migration:")
            for row in cur.fetchall():
                print(f"  {row[0]}: {row[1]}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
