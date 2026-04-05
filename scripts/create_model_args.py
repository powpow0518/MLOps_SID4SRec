"""
One-time script: generate model_args.pkl from best-known hyperparameters + data.

The serving container needs model_args.pkl (stored in model_volume) alongside
model weights because the model state_dict does not include args or lookup tables.

Run this from the project root before starting the serving container:
    python scripts/create_model_args.py

Output: models/model_args.pkl
"""
import sys
import os
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.data_generator import DataGenerator


def build_args():
    """Construct args namespace with Beauty best hyperparameters (from training/README.md)."""
    parser = argparse.ArgumentParser()

    # ── Fixed across datasets ────────────────────────────────────────────────
    parser.add_argument("--dataset",            default="Beauty")
    parser.add_argument("--data_path",          default="./data/raw/")
    parser.add_argument("--max_seq_length",     type=int,   default=50)
    parser.add_argument("--train_batch_size",   type=int,   default=256)
    parser.add_argument("--test_batch_size",    type=int,   default=512)
    parser.add_argument("--hidden_size",        type=int,   default=64)
    parser.add_argument("--n_layers",           type=int,   default=2)
    parser.add_argument("--n_heads",            type=int,   default=2)
    parser.add_argument("--inner_size",         type=int,   default=128)
    parser.add_argument("--learning_rate",      type=float, default=0.001)
    parser.add_argument("--weight_decay",       type=float, default=0.0)
    parser.add_argument("--diffusion_steps",    type=int,   default=1000)
    parser.add_argument("--hidden_act",         default="gelu")
    parser.add_argument("--layer_norm_eps",     type=float, default=1e-12)
    parser.add_argument("--initializer_range",  type=float, default=0.02)
    parser.add_argument("--sasrec_dropout_prob",type=float, default=0.5)
    parser.add_argument("--attn_dropout_prob",  type=float, default=0.5)
    parser.add_argument("--loss_type",          default="BPR")
    parser.add_argument("--seed",               type=int,   default=2024)

    # ── Diffusion params ─────────────────────────────────────────────────────
    parser.add_argument("--noise_schedule",         default="sqrt")
    parser.add_argument("--max_beta",               type=float, default=0.999)
    parser.add_argument("--inference_sampling_steps", type=int, default=5)
    parser.add_argument("--rescale_timesteps",      type=bool,  default=True)
    parser.add_argument("--predict_xstart",         type=bool,  default=True)
    parser.add_argument("--learn_sigma",            type=bool,  default=False)
    parser.add_argument("--sigma_small",            type=bool,  default=False)
    parser.add_argument("--use_kl",                 type=bool,  default=False)
    parser.add_argument("--rescale_learned_sigmas", type=bool,  default=False)
    parser.add_argument("--seq_len",                type=int,   default=50)
    parser.add_argument("--dropout",                type=float, default=0.1)
    parser.add_argument("--use_fp16",               type=bool,  default=False)
    parser.add_argument("--ema_rate",               type=float, default=1.0)
    parser.add_argument("--emb_scale_factor",       type=float, default=1.0)
    parser.add_argument("--gradient_clipping",      type=float, default=-1.0)
    parser.add_argument("--fp16_scale_growth",      type=float, default=0.001)
    parser.add_argument("--schedule_sampler",       default="lossaware")
    parser.add_argument("--resume_checkpoint",      default="none")
    parser.add_argument("--timestep_respacing",     default="")
    parser.add_argument("--use_plm_init",           default="no")
    parser.add_argument("--notes",                  default="folder-notes")
    parser.add_argument("--num_hidden_layers",      type=int,   default=2)
    parser.add_argument("--intermediate_size",      type=int,   default=128)
    parser.add_argument("--num_attention_heads",    type=int,   default=2)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.2)
    parser.add_argument("--max_position_embeddings", type=int, default=50)
    parser.add_argument("--max_relative_positions", type=int,  default=-1)
    parser.add_argument("--type_vocab_size",        type=int,   default=0)
    parser.add_argument("--hidden_dropout_prob",    type=float, default=0.2)
    parser.add_argument("--is_decoder",             type=bool,  default=False)
    parser.add_argument("--mask_type",              default="prob")
    parser.add_argument("--pad_token_id",           type=int,   default=0)
    parser.add_argument("--relative_attention",     type=bool,  default=True)
    parser.add_argument("--position_biased_input",  type=bool,  default=False)
    parser.add_argument("--mlm_probability_train",  type=float, default=0.1)
    parser.add_argument("--mlm_probability",        type=float, default=0.1)
    parser.add_argument("--batch_size",             type=int,   default=256)

    # ── Beauty best hyperparameters (training/README.md) ─────────────────────
    parser.add_argument("--temperature",  type=float, default=0.7)   # psi_seq
    parser.add_argument("--psi_seq",      type=float, default=0.7)
    parser.add_argument("--psi_item",     type=float, default=0.7)   # phi
    parser.add_argument("--beta",         type=float, default=0.1)
    parser.add_argument("--lambda_cl",    type=float, default=0.8)
    parser.add_argument("--alpha",        type=float, default=0.1)
    parser.add_argument("--gamma",        type=float, default=0.0)
    parser.add_argument("--item_temp",    type=float, default=0.1)

    # ── Misc ─────────────────────────────────────────────────────────────────
    parser.add_argument("--model_name",   default="diffsas")
    parser.add_argument("--model_idx",    default="0")
    parser.add_argument("--filter_num",   type=int, default=5)
    parser.add_argument("--epochs",       type=int, default=1000)
    parser.add_argument("--output_dir",   default="./models/")
    parser.add_argument("--check_path",   default="")
    parser.add_argument("--gpu_id",       type=int, default=0)
    parser.add_argument("--pos_att_type", default=["p2c", "c2p"])

    return parser.parse_args([])


def main():
    os.makedirs("models", exist_ok=True)

    print("Building args from best hyperparameters (Beauty dataset)...")
    args = build_args()

    print("Loading data to build lookup tables...")
    generator = DataGenerator(args)
    # DataGenerator.create_dataset() populates args with:
    #   item_size, category_lookup, brand_lookup, item_to_category,
    #   item_to_brand, category_items, brand_items, etc.
    # (lookup tensors are on CPU here; serving/main.py moves them to GPU at startup)

    # Remove training-only attrs that contain numpy/scipy objects not needed at serving time
    for attr in ("valid_rating_matrix", "test_rating_matrix", "items_feature"):
        if hasattr(args, attr):
            delattr(args, attr)

    output_path = os.path.join(args.output_dir, "model_args.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(args, f)

    print(f"Saved: {output_path}")
    print(f"  item_size      : {args.item_size}")
    print(f"  category_lookup: {args.category_lookup.shape}")
    print(f"  brand_lookup   : {args.brand_lookup.shape}")


if __name__ == "__main__":
    main()
