"""
Batch-generate user representation vectors and recommendation logs, then store in DB.

每個 user 的 representation = SID4SRec transformer 最後位置的輸出（dim=192）。
key: (user_id, model_version)，UPSERT 語意（retrain 後重跑會更新舊的）。
同時將 top-k 推薦結果寫入 recommendation_log，供 /explain 使用。

Cold-start: 訓練 snapshot 之後新增的 item（item_id > snapshot_max_item_id）
透過 serving.cold_start 的 substitute_map + synthetic embeddings 處理，
仍可出現在 top-k 結果裡。

Usage:
    python -m scripts.generate_user_representations
"""

import os
import pickle
from datetime import datetime, timezone

import psycopg2
import torch

from serving.cold_start import build_cold_start_data, run_inference
from training.sid4srec import SID4SRec

MODEL_PATH = os.getenv("MODEL_PATH", "/models/best_model.pt")
MODEL_ARGS_PATH = os.getenv("MODEL_ARGS_PATH", "/models/model_args.pkl")
DATABASE_URL = os.environ["DATABASE_URL"]
BATCH_SIZE = 256
TOP_K = 20


def load_model(device):
    with open(MODEL_ARGS_PATH, "rb") as f:
        args = pickle.load(f)
    args.category_lookup = args.category_lookup.to(device)
    args.brand_lookup = args.brand_lookup.to(device)

    model = SID4SRec(device, args)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, args


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model, args = load_model(device)
    max_len = args.max_seq_length
    snapshot_max_item_id = getattr(args, 'snapshot_max_item_id', 0)
    db2train = getattr(args, 'db2train', {})
    print(f"Snapshot max item_id: {snapshot_max_item_id}, vocab size: {len(db2train)}")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT model_version FROM model_version WHERE is_active = TRUE LIMIT 1"
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("No active model version found in DB. Run generate_embeddings first.")
            model_version = row[0]
            print(f"Model version: {model_version}")

            # 所有 user 的互動序列（DB item_ids，含 cold items）
            cur.execute("""
                SELECT user_id, array_agg(item_id ORDER BY timestamp ASC) AS item_sequence
                FROM interaction
                GROUP BY user_id
            """)
            users = cur.fetchall()
            print(f"Users to process: {len(users)}")

            # 預先取得 cold-start item rows（item_id > snapshot boundary）
            cur.execute("""
                SELECT item_id, category_id1, brand_id
                FROM item
                WHERE item_id > %s
                ORDER BY item_id
            """, (snapshot_max_item_id,))
            cold_rows = cur.fetchall()

        print(f"Cold-start items: {len(cold_rows)}")

        # 建立 cold-start substitute map + synthetic embeddings（一次性，所有 user 共用）
        substitute_map, cold_item_ids, cold_embeddings = build_cold_start_data(
            cold_rows, args, model, device
        )

        # Batch inference
        repr_results = []
        rec_results = []
        now = datetime.now(timezone.utc)

        for i in range(0, len(users), BATCH_SIZE):
            batch = users[i : i + BATCH_SIZE]

            for user_id, seq in batch:
                if not seq:
                    continue

                top_items, user_repr = run_inference(
                    seq, args, model, device,
                    top_k=TOP_K,
                    substitute_map=substitute_map,
                    cold_item_ids=cold_item_ids,
                    cold_embeddings=cold_embeddings,
                )

                array_literal = "{" + ",".join(str(x) for x in top_items) + "}"
                vector_str = "[" + ",".join(f"{v:.6f}" for v in user_repr) + "]"

                repr_results.append((user_id, model_version, vector_str))
                rec_results.append((user_id, array_literal, now))

            if (i // BATCH_SIZE + 1) % 10 == 0:
                processed = min(i + BATCH_SIZE, len(users))
                print(f"  {processed}/{len(users)} users processed")

        # Batch UPSERT user_representation
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO user_representation (user_id, model_version, representation)
                VALUES (%s, %s, %s::vector)
                ON CONFLICT (user_id, model_version) DO UPDATE
                    SET representation = EXCLUDED.representation,
                        created_at     = NOW()
                """,
                repr_results,
            )

        # Batch INSERT recommendation_log
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO recommendation_log (user_id, recommended_items, created_at)
                VALUES (%s, %s, %s)
                """,
                rec_results,
            )

        conn.commit()
        print(
            f"Done. Stored {len(repr_results)} user representations and "
            f"{len(rec_results)} recommendation logs for version '{model_version}'."
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
