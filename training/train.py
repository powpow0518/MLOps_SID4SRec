import argparse
import logging
import os
import pickle
import shutil
import torch
from data_pipeline.data_generator import DataGenerator
from training.trainer import Trainer
from training.utils import set_seed, check_path
from training.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    args = get_config()
    set_seed(args.seed)
    check_path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    args_str = f"{args.model_name}-{args.dataset}-{args.model_idx}"
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    data_generator = DataGenerator(args)

    # Save model_args.pkl so serving / inference scripts get the up-to-date vocab
    # (item_size, remap tables, lookup tensors) from this exact training run.
    # Use a shallow copy and drop heavy training-only attrs; keep `args` intact for trainer.
    model_args_path = os.path.join("/models", "model_args.pkl")
    args_for_pkl = argparse.Namespace(**vars(args))
    for attr in ("valid_rating_matrix", "test_rating_matrix", "items_feature"):
        if hasattr(args_for_pkl, attr):
            delattr(args_for_pkl, attr)
    with open(model_args_path, "wb") as f:
        pickle.dump(args_for_pkl, f)
    logger.info("Saved model_args.pkl → %s (item_size=%d)", model_args_path, args.item_size)

    trainer = Trainer(args, device, data_generator)
    trainer.train()

    # Publish the best checkpoint to the serving path so serving / generate_embeddings
    # / generate_user_representations all see the model from THIS run.
    # Trainer saves to args.checkpoint_path (./saved_models/<args_str>.pt); without this
    # copy, /models/best_model.pt would still be the previous run → vocab size mismatch.
    model_output_path = os.getenv("MODEL_OUTPUT_PATH", "/models/best_model.pt")
    if os.path.exists(args.checkpoint_path):
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        shutil.copy2(args.checkpoint_path, model_output_path)
        logger.info("Published checkpoint → %s", model_output_path)
    else:
        logger.warning("Checkpoint not found at %s — MODEL_OUTPUT_PATH not updated", args.checkpoint_path)

if __name__ == "__main__":
    main()
