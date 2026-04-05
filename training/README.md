# SID4SRec
**Sequence- and Item-Level Contrastive Learning via Diffusion-Based Data Augmentation for Sequential Recommendation**

SID4SRec is a sequential recommendation framework that integrates **Diffusion-based Data Augmentation** with **Dual-level Contrastive Learning** (Sequence and Item levels). It enhances item representations by incorporating rich context (Categories and Brands) and leverages the generative power of Diffusion Models to create high-quality augmented samples for self-supervised learning.

---

## Key Features

- **Diffusion-Based Augmentation**: Uses a Gaussian Diffusion process to generate semantically consistent augmented sequences (`aug_seq1`, `aug_seq2`) by denoising latent item representations.
- **Sequence-Level Contrastive Learning**: Implements InfoNCE loss on diffusion-augmented sequences to capture robust user intent.
- **Item-Level Contrastive Learning**: Leverages item metadata (Category/Brand) with an instance-weighting threshold ($\phi$) to filter false negatives and refine item embeddings.
- **Context-Aware Backbone**: A SASRec-style Transformer architecture that fuses Item ID, Category, and Brand embeddings into a unified representation.

## Environment Setup

- Python 3.8+
- PyTorch 2.0.0+
- Requirements: `pip install -r requirements.txt`

## Quick Start

To train and evaluate the model on a specific dataset (e.g., Beauty):

```bash
python main.py --dataset Beauty --model_name diffsas --gpu_id 0
```

### Key Hyperparameters
- `--alpha`: Weight for Sequence-level CL loss.
- `--lambda_cl`: Weight for Item-level CL loss.
- `--beta`: Weight for Diffusion NLL loss.
- `--phi`: Instance weighting threshold for Item-level CL (filtering false negatives).
- `--temperature`: Temperature for Sequence-level InfoNCE loss (ψ_seq).

### Best Hyperparameters per Dataset

Fixed across all datasets: `--max_seq_length 50 --train_batch_size 256 --hidden_size 64 --n_layers 2 --n_heads 2 --learning_rate 0.001 --diffusion_steps 1000`

| Parameter | Beauty | Sports_and_Outdoors | Toys_and_Games |
|-----------|--------|---------------------|----------------|
| `--temperature` (ψ_seq) | 0.7 | 0.5 | 0.1 |
| `--phi` (ψ_item) | 0.7 | 0.5 | 0.7 |
| `--beta` | 0.1 | 0.5 | 0.1 |
| `--lambda_cl` | 0.8 | 0.1 | 0.6 |
| `--alpha` | 0.1 | 0.1 | 0.1 |

Example (Beauty best settings):
```bash
python main.py --dataset Beauty --gpu_id 0 --temperature 0.7 --phi 0.7 --beta 0.1 --lambda_cl 0.8 --alpha 0.1
```

## Project Structure

- `main.py`: Entry point for training and evaluation.
- `models/sid4srec.py`: Core implementation of the SID4SRec architecture.
- `models/gaussian_diffusion.py`: Diffusion process and data augmentation logic.
- `trainers/trainer.py`: Training loops and loss computation.
- `data_generators/`: Data preprocessing and context (Category/Brand) injection.
- `configs/`: Dataset-specific hyperparameter configurations.
"# SID4SRec" 
