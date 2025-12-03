# LiTE (Lightweight Inception Time for Time Series)

Reference: M. Devanne, M. Ismail-Fawaz et al., “LiTE: Lightweight Inception Time for Time Series,” DSAA 2023. Paper: https://maxime-devanne.com/delegation/publis/ismail-fawaz_dsaa2023.pdf

This repo provides a PyTorch implementation of LiTE and simple training/inference scripts on UCR datasets (example: `PhalangesOutlinesCorrect` in `data/`).

## Quickstart
1) Create/activate venv  
   ```bash
   cd /Users/quentinpour/python/LiTE_pytorch
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install torch numpy scikit-learn tqdm matplotlib pandas
   ```
2) Train on the provided dataset  
   ```bash
   python train.py --dataset PhalangesOutlinesCorrect --epochs 20 --batch-size 32
   ```
3) Inspect quick predictions (printed after training) and optional model save (`--save-path model.pth`).

## Key Files
- `src/lite.py`: LiTE model.
- `src/layers/*`: Inception, hybrid, and FCN modules.
- `train.py`: Data loading, training loop, and simple inference.
- `utils/utils.py`, `utils/trainer.py`: datasets, normalization, training helpers.

## Notes
- `train.py` expects UCR-format files under `data/<dataset>/<dataset>_TRAIN.txt` and `_TEST.txt`.
- Loss is cross-entropy on the model’s softmax output (`CrossEntropyFromProbs` wrapper). Adjust flags (filters, dilation) via CLI args.***
