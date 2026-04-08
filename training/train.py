import logging
import os
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
    trainer = Trainer(args, device, data_generator)
    trainer.train()

if __name__ == "__main__":
    main()
