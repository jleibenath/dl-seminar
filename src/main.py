import torch
import torch.nn as nn
import torch.optim as optim
from load_data import *
from SimpleNet import SimpleNet


def train_model(model, device, criterion, optimizer, train_loader, num_epochs=10):
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
        print(f"Epoche {epoch + 1}/{num_epochs}, Verlust: {running_loss / len(train_loader.dataset):.4f}")


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
    print(f"Testgenauigkeit: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    train_loader, test_loader = get_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    weights = torch.tensor([661/139, 1.0], device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    train_model(model, device, criterion, optimizer, train_loader)
    test_model(model, device, test_loader)
