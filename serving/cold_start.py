"""
Shared cold-start handling and inference logic for SID4SRec.

Used by both:
  - serving/main.py  (online, per-request)
  - scripts/generate_user_representations.py  (batch, post-retrain)

Cold-start definition: a DB item_id that was NOT in the training snapshot
(i.e., item_id > args.snapshot_max_item_id, or not present in args.db2train).

Remap convention:
  args.db2train  : dict[int, int]   db_id -> train_id (1-indexed, consecutive)
  args.train2db  : list[int]        train_id -> db_id;  train2db[0] = 0 (padding)
  args.snapshot_max_item_id : int   highest db item_id included in training vocab
"""

import random
import torch


def build_cold_start_data(cold_item_rows, args, model, device):
    """Compute substitute map and synthetic embeddings for cold-start items.

    Parameters
    ----------
    cold_item_rows : list of (item_id, category_id1, brand_id)
        Items that are NOT in the training vocab (fetched by the caller).
    args : namespace
        model_args with category_lookup, brand_lookup, item_size, db2train, train2db.
    model : SID4SRec
    device : torch.device

    Returns
    -------
    substitute_map : dict[int, int]
        cold_db_id -> proxy_train_id (used to rewrite input sequences safely).
    cold_item_ids : list[int]
        cold DB item ids, aligned with cold_embeddings rows.
    cold_embeddings : Tensor | None
        [N_cold, emb_dim] synthetic embeddings, or None if no cold items.
    """
    if not cold_item_rows:
        return {}, [], None

    item_size = args.item_size
    cat_lookup = args.category_lookup.to(device)   # [item_size] indexed by train_id
    brand_lookup = args.brand_lookup.to(device)    # [item_size] indexed by train_id

    with torch.no_grad():
        in_vocab_emb = model.items_emb()  # [item_size, emb_dim]

    substitute_map: dict = {}
    cold_item_ids: list = []
    cold_emb_list = []

    for cold_db_id, cold_cat, cold_brand in cold_item_rows:
        cat_t = torch.tensor(cold_cat or 0, device=device)
        brand_t = torch.tensor(cold_brand or 0, device=device)

        match = torch.where((cat_lookup == cat_t) & (brand_lookup == brand_t))[0]
        if match.numel() == 0:
            match = torch.where((cat_lookup == cat_t) | (brand_lookup == brand_t))[0]
        if match.numel() == 0:
            match = torch.tensor([random.randint(1, item_size - 1)], device=device)

        avg_emb = in_vocab_emb[match].mean(dim=0)  # [emb_dim]
        substitute_map[int(cold_db_id)] = int(match[0].item())  # proxy train_id
        cold_item_ids.append(int(cold_db_id))
        cold_emb_list.append(avg_emb)

    cold_embeddings = torch.stack(cold_emb_list, dim=0)  # [N_cold, emb_dim]
    return substitute_map, cold_item_ids, cold_embeddings


def run_inference(
    item_sequence_db_ids: list,
    args,
    model,
    device,
    top_k: int = 20,
    substitute_map: dict | None = None,
    cold_item_ids: list | None = None,
    cold_embeddings=None,
):
    """Run inference for a single user and return (top_k_db_ids, user_repr_list).

    Parameters
    ----------
    item_sequence_db_ids : list[int]
        Raw DB item ids from the interaction table (may include cold items).
    args : namespace
        model_args; must have db2train, train2db, item_size, max_seq_length.
    model : SID4SRec
    device : torch.device
    top_k : int
    substitute_map : dict[cold_db_id -> proxy_train_id]
    cold_item_ids : list[int]  cold DB ids aligned with cold_embeddings rows
    cold_embeddings : Tensor | None

    Returns
    -------
    top_items_db_ids : list[int]   top-k recommended DB item ids
    user_repr : list[float]        192-dim user representation vector
    """
    db2train = args.db2train
    train2db = args.train2db
    item_size = args.item_size
    substitute_map = substitute_map or {}

    # Rewrite sequence to train_ids; cold items use proxy train_id from substitute_map
    safe_seq = []
    for db_id in item_sequence_db_ids:
        if db_id in db2train:
            safe_seq.append(db2train[db_id])
        else:
            safe_seq.append(substitute_map.get(db_id, 0))  # 0 = padding fallback

    max_len = args.max_seq_length
    seq = safe_seq[-max_len:]
    padded = [0] * (max_len - len(seq)) + seq
    input_tensor = torch.tensor([padded], dtype=torch.long).to(device)

    with torch.no_grad():
        user_repr = model.get_user_representation(input_tensor)          # [1, emb_dim]
        in_vocab_emb = model.items_emb()                                 # [item_size, emb_dim]
        in_vocab_scores = torch.matmul(user_repr, in_vocab_emb.T)[0].cpu()  # [item_size]

        if cold_embeddings is not None and cold_embeddings.numel() > 0:
            cold_scores = torch.matmul(user_repr, cold_embeddings.T)[0].cpu()
        else:
            cold_scores = None

    # Mask padding slot
    in_vocab_scores[0] = -1e9

    # Mask already-seen in-vocab items (use train_id as index into score vector)
    for db_id in item_sequence_db_ids:
        train_id = db2train.get(db_id)
        if train_id and 0 < train_id < item_size:
            in_vocab_scores[train_id] = -1e9

    # Combine in-vocab + cold scores; map result indices → DB ids
    if cold_scores is not None and cold_item_ids:
        seen_set = set(item_sequence_db_ids)
        for i, cid in enumerate(cold_item_ids):
            if cid in seen_set:
                cold_scores[i] = -1e9
        all_scores = torch.cat([in_vocab_scores, cold_scores])
        # train2db maps train_id→db_id for in-vocab portion; cold portion is already db_ids
        all_ids = list(train2db) + cold_item_ids
    else:
        all_scores = in_vocab_scores
        all_ids = list(train2db)

    top_idx = torch.argsort(all_scores, descending=True)[:top_k].tolist()
    top_items_db_ids = [all_ids[i] for i in top_idx]
    return top_items_db_ids, user_repr[0].cpu().tolist()
