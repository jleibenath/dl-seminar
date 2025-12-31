from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
from load_data import *
from SimpleNet import SimpleNet
from sklearn.model_selection import StratifiedKFold


def train_model(model, device, criterion, optimizer, train_loader, num_epochs=30):
    for epoch in range(num_epochs):
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
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader.dataset):.4f}")


def test_model(model, device, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    # TODO: add confusion matrix
    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%\n")
    return accuracy


def train_and_eval():
    train_loader, test_loader = get_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    train_model(model, device, criterion, optimizer, train_loader)
    test_model(model, device, test_loader)


def val_train_and_eval(folds=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    train_set, test_set = get_seq_datasets()
    train_data, train_targets = train_set.tensors
    results = []
    kfold = StratifiedKFold(n_splits=folds, shuffle=True)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_data, train_targets)):

        train_sub = Subset(train_set, train_ids)
        val_sub = Subset(train_set, val_ids)

        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

        model = SimpleNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(model, device, criterion, optimizer, train_loader)
        results.append(test_model(model, device, val_loader))
    print(results)
    print(f"Average: {sum(results) / len(results)}")


if __name__ == "__main__":
    # val_train_and_eval(5)
    train_and_eval()
