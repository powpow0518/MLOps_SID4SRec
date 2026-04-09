"""
Generate item embeddings from trained model and store in DB.

Usage:
    python -m scripts.generate_embeddings
    python -m scripts.generate_embeddings --model-version v2
"""

import argparse
import os
import pickle
from datetime import datetime

import psycopg2
import torch

from training.sid4srec import SID4SRec

MODEL_PATH = os.getenv("MODEL_PATH", "/models/best_model.pt")
MODEL_ARGS_PATH = os.getenv("MODEL_ARGS_PATH", "/models/model_args.pkl")
DATABASE_URL = os.environ["DATABASE_URL"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-version",
        default=None,
        help="Model version string (default: auto timestamp)",
    )
    return parser.parse_args()


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


def generate_embeddings(model):
    """Return item embeddings as numpy array, shape (item_size, 192)."""
    with torch.no_grad():
        embeddings = model.items_emb()  # (item_size, hidden_size * 3)
    return embeddings.cpu().numpy()


def store_embeddings(embeddings, model_version: str, args=None):
    """Insert embeddings into item_embedding table, replacing old version."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn, conn.cursor() as cur:
            # Register model version
            cur.execute(
                """
                    INSERT INTO model_version (model_version, is_active, note)
                    VALUES (%s, FALSE, 'auto-generated embeddings')
                    ON CONFLICT (model_version) DO NOTHING
                    """,
                (model_version,),
            )

            # train2db[train_id] = db_id；用 remap 把 train_id 還原成 DB item_id
            train2db = getattr(args, 'train2db', []) if args else []
            cur.execute("SELECT item_id FROM item")
            valid_item_ids = {row[0] for row in cur.fetchall()}

            batch = []
            for train_id, embedding in enumerate(embeddings):
                if train_id == 0:
                    continue  # skip padding
                if train2db:
                    db_id = train2db[train_id] if train_id < len(train2db) else None
                else:
                    db_id = train_id  # fallback: no remap (old model_args.pkl)
                if db_id and db_id in valid_item_ids:
                    batch.append((int(db_id), model_version, embedding.tolist()))

            cur.executemany(
                """
                    INSERT INTO item_embedding (item_id, model_version, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (item_id, model_version) DO UPDATE
                        SET embedding = EXCLUDED.embedding
                    """,
                batch,
            )

            # Mark this version as active, deactivate others
            cur.execute("UPDATE model_version SET is_active = FALSE")
            cur.execute(
                "UPDATE model_version SET is_active = TRUE WHERE model_version = %s",
                (model_version,),
            )

        print(f"Stored {len(batch)} embeddings for version '{model_version}'")
    finally:
        conn.close()


def main():
    args = get_args()
    model_version = args.model_version or datetime.utcnow().strftime("v%Y%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model, model_args = load_model(device)

    print("Generating embeddings...")
    embeddings = generate_embeddings(model)
    print(f"Embedding shape: {embeddings.shape}")

    print("Storing embeddings in DB...")
    store_embeddings(embeddings, model_version, args=model_args)
    print("Done.")


if __name__ == "__main__":
    main()
