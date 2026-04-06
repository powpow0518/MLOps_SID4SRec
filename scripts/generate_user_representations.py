"""
Batch-generate user representation vectors and store in DB.

每個 user 的 representation = SID4SRec transformer 最後位置的輸出（dim=192）。
key: (user_id, model_version)，UPSERT 語意（retrain 後重跑會更新舊的）。

Usage:
    python -m scripts.generate_user_representations
"""

import os
import pickle

import psycopg2
import torch

from training.sid4srec import SID4SRec

MODEL_PATH = os.getenv("MODEL_PATH", "/models/best_model.pt")
MODEL_ARGS_PATH = os.getenv("MODEL_ARGS_PATH", "/models/model_args.pkl")
DATABASE_URL = os.environ["DATABASE_URL"]
BATCH_SIZE = 256


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

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            # 取得目前 active model version（由 generate_embeddings 設定）
            cur.execute(
                "SELECT model_version FROM model_version WHERE is_active = TRUE LIMIT 1"
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("No active model version found in DB. Run generate_embeddings first.")
            model_version = row[0]
            print(f"Model version: {model_version}")

            # 取得所有 user 的互動序列（依時間排序）
            cur.execute("""
                SELECT user_id, array_agg(item_id ORDER BY timestamp ASC) AS item_sequence
                FROM interaction
                GROUP BY user_id
            """)
            users = cur.fetchall()
            print(f"Users to process: {len(users)}")

        # Batch inference
        results = []
        with torch.no_grad():
            for i in range(0, len(users), BATCH_SIZE):
                batch = users[i : i + BATCH_SIZE]
                user_ids = []
                padded_seqs = []

                for user_id, seq in batch:
                    seq = seq[-max_len:]
                    padded = [0] * (max_len - len(seq)) + seq
                    user_ids.append(user_id)
                    padded_seqs.append(padded)

                input_tensor = torch.tensor(padded_seqs, dtype=torch.long).to(device)
                repr_vectors = model.get_user_representation(input_tensor).cpu().numpy()

                for user_id, repr_vec in zip(user_ids, repr_vectors):
                    vector_str = "[" + ",".join(str(float(v)) for v in repr_vec) + "]"
                    results.append((user_id, model_version, vector_str))

                if (i // BATCH_SIZE + 1) % 10 == 0:
                    print(f"  {i + len(batch)}/{len(users)} users processed")

        # Batch UPSERT
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO user_representation (user_id, model_version, representation)
                VALUES (%s, %s, %s::vector)
                ON CONFLICT (user_id, model_version) DO UPDATE
                    SET representation = EXCLUDED.representation,
                        created_at     = NOW()
                """,
                results,
            )
        conn.commit()
        print(f"Done. Stored {len(results)} user representations for version '{model_version}'.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
