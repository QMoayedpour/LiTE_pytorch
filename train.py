import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from src.lite import LITE
from utils.utils import load_data, znormalisation, SimpleDataset
from utils.trainer import train_model, evaluate_model


class CrossEntropyFromProbs(nn.Module):
    """Cross-entropy that accepts probabilities (softmax output) instead of logits."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.loss_fn = nn.NLLLoss()

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log(torch.clamp(probs, min=self.eps))
        return self.loss_fn(log_probs, targets)


def build_dataloaders(file_name: str, batch_size: int, device: torch.device):
    xtrain, ytrain, xtest, ytest = load_data(file_name, folder_path="data/")

    print(xtrain.shape)
    xtrain = znormalisation(xtrain)
    xtest = znormalisation(xtest)

    encoder = LabelEncoder()
    ytrain_enc = encoder.fit_transform(ytrain)
    ytest_enc = encoder.transform(ytest)

    train_dataset = SimpleDataset(xtrain, ytrain_enc, seq_len=0, device=device)
    test_dataset = SimpleDataset(xtest, ytest_enc, seq_len=0, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    n_classes = len(encoder.classes_)
    return train_loader, test_loader, n_classes, encoder


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, n_classes, encoder = build_dataloaders(
        file_name=args.dataset, batch_size=args.batch_size, device=device
    )

    model = LITE(
        n_classes=n_classes,
        n_filters=args.n_filters,
        kernel_size=args.kernel_size,
        use_custom_filters=args.use_custom_filters,
        use_dilation=not args.no_dilation,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = CrossEntropyFromProbs()

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        test_interval=args.test_interval,
        patience=args.patience,
        verbose=True,
    )

    evaluate_model(model, test_loader, criterion, device=device)

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu()
            for pred, label in zip(preds.tolist(), labels.tolist()):
                print(
                    f"Pred: {encoder.classes_[pred]} - True: {encoder.classes_[label]}"
                )
            break

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LiTE on a UCR dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="PhalangesOutlinesCorrect",
        help="Dataset name under data/",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--test-interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-filters", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=41)
    parser.add_argument(
        "--use-custom-filters",
        action="store_true",
        help="Enable hybrid filters in InceptionModule",
    )
    parser.add_argument(
        "--no-dilation", action="store_true", help="Disable dilation in FCN modules"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Optional path to save trained weights",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
