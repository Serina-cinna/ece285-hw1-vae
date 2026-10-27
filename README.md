# ECE 285 Homework 1: Chest X-ray VAE

This package contains the report source, training and evaluation code, and the results from the selected 128 x 128 run (seed 43). One unconditional VAE is trained on pooled NORMAL and PNEUMONIA images.

## Dataset

The assigned dataset is the [Kaggle Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The experiment used the public [CC BY 4.0 mirror](https://huggingface.co/datasets/ahulikal/chest-xray-pneumonia-mirror), which states that the original files and train/validation/test split are preserved. 

The 5,216 images in the original `train` folder were deterministically split into 4,694 training and 522 validation images. The original 624-image test folder was held out until final evaluation. Labels are ignored. 

## Environment and run

Install the dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train the selected configuration:

```bash
python code/train.py \
  --data data/chest_xray \
  --output outputs/seed43 \
  --epochs 150 --patience 18 --batch-size 64 \
  --learning-rate 0.0003 --latent-dim 128 \
  --image-size 128 --seed 43 --split-seed 42
```

Evaluate the best validation checkpoint on the original test split:

```bash
python code/evaluate.py \
  --data data/chest_xray \
  --output outputs/seed43 \
  --batch-size 64 --seed 2026
```

The training script saves the best and latest checkpoints, loss history, and a loss curve. The evaluation script saves IS, FID, and 64 generated prior samples. 

## Report and selected results

- PDF report: `report.pdf`
- LaTeX source: `report/latex/main.tex`

- Selected run outputs: `results/seed43/`
- Validation-only model selection summary: `results/model_selection.csv`
- Best validation checkpoint: epoch 120; validation negative ELBO 9,426.60 per image.
- Test evaluation: 624 generated and 624 real images; IS 1.926 and FID 0.162.

Model checkpoints and the dataset are excluded from this package to keep it small; training recreates the checkpoints.

