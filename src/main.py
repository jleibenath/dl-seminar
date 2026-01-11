import torch
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
from load_data import *
from SimpleNet import SimpleNet
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt


def f1_auc(logits, target):
    sm = nn.Softmax(dim=1)
    probs = sm(logits)
    pos_scores = probs[:, 1]
    y_pred = torch.argmax(logits, dim=1)
    target_np = target.numpy()
    y_pred_np = y_pred.numpy()
    pos_scores_np = pos_scores.numpy()
    f1 = f1_score(target_np, y_pred_np)
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
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    loss = running_loss / len(train_loader.dataset)
    return loss


def test_model(model, device, test_loader, criterion=None):
    model.eval()
    correct, total = 0, 0
    pred_ones = 0
    all_logits, all_targets = [], []
    loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            if criterion:
                curr_loss = criterion(outputs, y_batch).detach().cpu()
                loss += curr_loss * y_batch.size(0)
            all_logits.append(outputs.detach().cpu())
            all_targets.append(y_batch.cpu())
            _, predicted = torch.max(outputs, 1)
            pred_ones += predicted.sum()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = 100 * correct / total
    # print(f"predicted {pred_ones} 1s")
    print(f"Test accuracy: {accuracy:.2f}%")
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    f1, auc = f1_auc(all_logits, all_targets)
    loss = loss/len(test_loader.dataset)
    return accuracy, f1, auc, loss


def train_and_eval(runs=1, plot=False, path=None):
    train_loader, test_loader = get_dataloader()
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    running_acc = 0
    for i in range(runs):
        print(f"RUN {i+1}/{runs}\n")
        model = SimpleNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        losses = train_model(model, device, criterion, optimizer, train_loader)
        plt.plot(losses)
        cur_acc, _, _, _ = test_model(model, device, test_loader)
        running_acc += cur_acc
        if path:
            torch.save(model.state_dict(), path)
    if plot:
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("loss_plot.pdf", bbox_inches="tight")
        plt.close()
    return running_acc / runs


def val_train_and_eval(folds=2, epochs=30, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    train_set, test_set = get_seq_datasets()
    train_data, train_targets = train_set.tensors
    accuracies, f1_scores, auc_scores, val_losses, train_losses = [], [], [], [], []
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
    return accuracies, f1_scores, auc_scores, val_losses, train_losses


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


def main():
    accuracies, f1_scores, auc_scores, val_losses, train_losses = val_train_and_eval(5, lr=0.002)
    plot_losses(train_losses, val_losses)
    print(accuracies)
    print(f1_scores)
    print(auc_scores)
    print(f"\nAverages:")
    print(f"Validation Accuracy: {sum(accuracies)/len(accuracies):.2f}%")
    print(f"F1 Score: {sum(f1_scores)/len(f1_scores):.2f}")
    print(f"AUC: {sum(auc_scores)/len(auc_scores):.2f}")


if __name__ == "__main__":
    main()
