import torch
from torch.nn.functional import sigmoid
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
from load_data import *
from SimpleNet import SimpleNet
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


def f1_auc(logits, target):
    if len(logits.shape) == 2:
        sm = nn.Softmax(dim=1)
        probs = sm(logits)
        pos_scores = probs[:, 1]
        y_pred = torch.argmax(logits, dim=1)
    else:
        sm = nn.Sigmoid()
        probs = sm(logits)
        pos_scores = probs
        y_pred = (probs >= 0.5).int()
    target_np = target.numpy()
    y_pred_np = y_pred.numpy()
    pos_scores_np = pos_scores.numpy()
    f1 = f1_score(target_np, y_pred_np, pos_label=0)
    auc = roc_auc_score(target_np, pos_scores_np)
    return f1, auc


def train_model(model, device, criterion, optimizer, train_loader, num_epochs=30):
    losses = []
    for epoch in range(num_epochs):
        loss = train_epoch(model, device, criterion, optimizer, train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
        losses.append(loss)
    return losses


def train_epoch(model, device, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        # outputs = model(X_batch).squeeze(1)
        loss = criterion(outputs, y_batch)
        # loss = criterion(outputs, y_batch.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    loss = running_loss / len(train_loader.dataset)
    return loss


def run_tests(model, device, test_loader, criterion=None, threshold=0.5):
    model.eval()
    correct, total = 0, 0
    pred_ones = 0
    all_logits, all_targets = [], []
    loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            # outputs = model(X_batch).squeeze(1)
            if criterion:
                curr_loss = criterion(outputs, y_batch).detach().cpu()
                # curr_loss = criterion(outputs, y_batch.float()).detach().cpu()
                loss += curr_loss * y_batch.size(0)
            all_logits.append(outputs.detach().cpu())
            all_targets.append(y_batch.cpu())
            predicted = outputs[:, 1] >= threshold
            # predicted = (outputs >= threshold)
            pred_ones += predicted.sum()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct, total, pred_ones, all_logits, all_targets, loss


def test_model(model, device, test_loader, criterion=None, threshold=0.5):
    model.eval()
    correct, total, pred_ones, all_logits, all_targets, loss = run_tests(model, device, test_loader, criterion, threshold)
    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    f1, auc = f1_auc(all_logits, all_targets)
    loss = loss/len(test_loader.dataset)
    return accuracy, f1, auc, loss


def val_train(folds=5, epochs=30, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    train_set, test_set = get_seq_datasets()
    train_data, train_targets = train_set.tensors
    accuracies, f1_scores, auc_scores, val_losses, train_losses = [], [], [], [], []
    thresholds = []
    kfold = StratifiedKFold(n_splits=folds, shuffle=True)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_data, train_targets)):
        print(f"\nSTART Fold {fold+1}/{folds}")

        train_sub = Subset(train_set, train_ids)
        val_sub = Subset(train_set, val_ids)

        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

        model = SimpleNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        cur_train_losses = []
        cur_val_losses = []

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}:")
            train_loss = train_epoch(model, device, criterion, optimizer, train_loader)
            accuracy, f1, auc, val_loss = test_model(model, device, val_loader, criterion)
            if epoch == epochs-1:
                accuracies.append(accuracy)
                f1_scores.append(f1)
                auc_scores.append(auc)
            cur_val_losses.append(val_loss)
            cur_train_losses.append(train_loss)
        train_losses.append(cur_train_losses)
        val_losses.append(cur_val_losses)
        thresholds.append(float(get_best_threshhold(model, device, val_loader, criterion)))
    return accuracies, f1_scores, auc_scores, val_losses, train_losses, thresholds


def plot_losses(train_losses, val_losses=None):
    for losses in train_losses:
        plt.plot(losses, color='red')
    if val_losses:
        for losses in val_losses:
            plt.plot(losses, color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_plot.pdf", bbox_inches="tight")
    plt.close()


def train_and_eval():
    accuracies, f1_scores, auc_scores, val_losses, train_losses, thresholds = val_train(5, lr=0.005)
    plot_losses(train_losses, val_losses)
    print(accuracies)
    print(f1_scores)
    print(auc_scores)
    print(thresholds)
    print(f"\nAverages:")
    print(f"Validation Accuracy: {sum(accuracies) / len(accuracies):.2f}%")
    print(f"F1 Score: {sum(f1_scores) / len(f1_scores):.3f}")
    print(f"AUC: {sum(auc_scores) / len(auc_scores):.3f}")
    print(f"average best threshold: {sum(thresholds)/len(thresholds):.3f}")


def get_best_threshhold(model, device, test_loader, criterion):
    correct, total, pred_ones, all_logits, all_targets, loss = run_tests(model, device, test_loader, criterion)
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    if len(all_logits.shape) == 2:
        sm = nn.Softmax(dim=1)
        probs = sm(all_logits)
        pos_scores = probs[:, 1]
    else:
        sm = nn.Sigmoid()
        probs = sm(all_logits)
        pos_scores = probs
    target_np = all_targets.numpy()
    pos_scores_np = pos_scores.numpy()
    fpr, tpr, thresholds = roc_curve(target_np, pos_scores_np)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def train_final():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    train_loader, test_loader = get_dataloader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 0.005)
    train_model(model, device, criterion, optimizer, train_loader)
    accuracy, f1, auc, val_loss = test_model(model, device, test_loader, criterion, threshold=0.8)
    print(f"\nScores:")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")



if __name__ == "__main__":
    train_final()
