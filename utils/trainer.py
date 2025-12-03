import torch
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import classification_report


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs=5,
    device="cpu",
    test_interval=10,
    print_result=True,
    patience=2,
    verbose=True,
):
    model.train()
    pbar_epoch = trange(num_epochs, desc="Epochs", disable=not verbose)
    best_accuracy = 0.0
    epochs_no_improve = 0
    for epoch in pbar_epoch:
        running_loss = 0.0
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            leave=False,
            disable=not verbose,
        ) as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                target_labels = labels
                if labels.ndim > 1:
                    target_labels = torch.argmax(labels, dim=1)
                loss = criterion(outputs, target_labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                pbar.set_postfix(
                    {"loss": running_loss / ((pbar.n + 1) * inputs.size(0))}
                )
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader.dataset)
        pbar_epoch.set_postfix({"epoch_loss": epoch_loss})

        if (epoch + 1) % test_interval == 0:
            model.eval()
            correct = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.append(preds.cpu().detach().numpy())
                    all_labels.append(labels.cpu().detach().numpy())
                    labs = torch.argmax(labels, dim=1) if labels.ndim > 1 else labels
                    correct += torch.sum(preds == labs)
            accuracy = correct.double() / len(test_loader.dataset)
            if print_result and verbose:
                all_labels = np.concatenate(all_labels, axis=0)
                all_preds = np.concatenate(all_preds, axis=0)
                print(classification_report(all_labels, all_preds))
            if verbose:
                print(f"Test Accuracy after {epoch + 1} epochs: {accuracy:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_no_improve = 0
                torch.save(model.state_dict(), "model.pth")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
            model.train()
    if verbose:
        print("Training complete!")


def evaluate_model(model, test_loader, criterion, device="cpu"):
    model.eval()
    test_loss = 0.0
    correct = 0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            target_labels = labels
            if labels.ndim > 1:
                target_labels = torch.argmax(labels, dim=1)
            loss = criterion(outputs, target_labels)
            test_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            labs = torch.argmax(labels, dim=1) if labels.ndim > 1 else labels
            correct += torch.sum(preds == labs)

            all_scores.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = correct.double() / len(test_loader.dataset)

    tqdm.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_scores, all_labels


class Learner:
    def __init__(
        self,
        model,
        dataset,
        x,
        loss,
        seq_len=336,
        batch_size=128,
        lr=0.0001,
        epochs=100,
        target_window=96,
        d_model=16,
        adjust_lr=True,
        adjust_factor=0.0,
        patiente=5,
        output_path="",
        device="cpu",
        univariate="",
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        if univariate == "":
            univariate = 1 if len(x.shape) == 1 else 0
        # univariate = 1
        train_dataset = dataset(
            x,
            mode="train",
            univariate=univariate,
            seq_len=seq_len,
            target_window=target_window,
        )
        valid_dataset = dataset(
            x,
            mode="val",
            univariate=univariate,
            seq_len=seq_len,
            target_window=target_window,
        )
        test_dataset = dataset(
            x,
            mode="test",
            univariate=univariate,
            seq_len=seq_len,
            target_window=target_window,
        )
        self.train_datalen = len(train_dataset)
        self.valid_datalen = len(valid_dataset)
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = loss
        self.epochs = epochs
        self.target_window = target_window
        self.best_weight = self.model.state_dict()
        self.d_model = d_model
        self.adjust_lr = adjust_lr
        self.adjust_factor = adjust_factor
        self.patience = 5
        self.output_path = output_path

    def adjust_learning_rate(self, steps, warmup_step=300, printout=False):
        if steps ** (-0.5) < steps * (warmup_step**-1.5):
            lr_adjust = (16**-0.5) * (steps**-0.5) * self.adjust_factor
        else:
            lr_adjust = (16**-0.5) * (steps * (warmup_step**-1.5)) * self.adjust_factor

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_adjust
        if printout:
            print("Updating learning rate to {}".format(lr_adjust))
        return

    def train(self):
        best_valid_loss = np.inf
        train_history = []
        valid_history = []
        count = 0
        train_steps = 1
        if self.adjust_lr:
            self.adjust_learning_rate(train_steps)
        bar = range(self.epochs)
        with tqdm(bar) as pbar:
            for epoch in pbar:
                # train
                self.model.train()
                iter_count = 0
                total_loss = 0

                for train_x, train_y in self.train_dataloader:
                    train_x = train_x.to(self.device)
                    train_y = train_y.to(self.device)

                    pred_y = self.model(train_x)
                    loss = self.loss(pred_y, train_y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    iter_count += 1
                    train_steps += 1
                if self.adjust_lr:
                    self.adjust_learning_rate(train_steps)

                # valid
                self.model.eval()
                valid_iter_count = 0
                valid_total_loss = 0
                with torch.no_grad():
                    for valid_x, valid_y in self.valid_dataloader:
                        valid_x = valid_x.to(self.device)
                        valid_y = valid_y.to(self.device)
                        pred_y = self.model(valid_x)
                        loss = self.loss(pred_y, valid_y)
                        valid_total_loss += loss.item()
                        valid_iter_count += 1

                total_loss /= iter_count
                valid_total_loss /= valid_iter_count
                text = "epoch: {} MSE loss: {:.4f} MSE valid loss: {:.4f}".format(
                    epoch, total_loss, valid_total_loss
                )
                pbar.set_postfix(exemple=text)
                if best_valid_loss >= valid_total_loss:
                    self.best_weight = self.model.state_dict()
                    best_valid_loss = valid_total_loss
                    # path = self.output_path + "patchTST_model.pth"
                    # torch.save(self.model.state_dict(), path)
                    count = 0
                else:
                    count += 1
                    if count >= self.patience:
                        train_history.append(total_loss)
                        valid_history.append(valid_total_loss)
                        print(
                            f"Training cancelled because no improvement since {self.patience} epochs"
                        )
                        return train_history, valid_history
                train_history.append(total_loss)
                valid_history.append(valid_total_loss)
        return train_history, valid_history

    def test(self):
        self.model.load_state_dict(self.best_weight)
        self.model.eval()
        iter_count = 0
        total_loss = 0
        with torch.no_grad():
            for test_x, test_y in self.test_dataloader:
                test_x = test_x.to(self.device)
                test_y = test_y.to(self.device)
                pred_y = self.model(test_x)
                loss = self.loss(pred_y, test_y)
                total_loss += loss.item()
                iter_count += 1
        total_loss /= iter_count
        print("MSE test loss: {:.4f}".format(total_loss))
