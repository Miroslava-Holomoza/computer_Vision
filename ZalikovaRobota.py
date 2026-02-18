import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

BATCH_SIZE = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # стандартна нормалізація для MNIST
])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)                            

        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
print(model)

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"Epoch {epoch}: train loss={avg_loss:.4f}, train acc={acc:.4f}")

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"          test loss={avg_loss:.4f}, test acc={acc:.4f}")
    return avg_loss, acc

if __name__ == '__main__':
    EPOCHS = 5
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, epoch)
        evaluate(model, test_loader)

    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Saved: mnist_cnn.pth")

@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    all_images = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
        all_images.append(x.cpu())  

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_images = torch.cat(all_images).numpy()  
    return all_images, all_targets, all_preds

def make_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def plot_confusion_matrix(cm, class_names=None, normalize=False, title="Confusion Matrix"):
    import numpy as np
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    if normalize:
        cm_to_plot = cm.astype(np.float64)
        row_sums = cm_to_plot.sum(axis=1, keepdims=True)
        cm_to_plot = np.divide(cm_to_plot, row_sums, out=np.zeros_like(cm_to_plot), where=row_sums != 0)
        fmt = ".2f"
    else:
        cm_to_plot = cm.astype(np.int64)
        fmt = "d"

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_to_plot, interpolation="nearest")
    plt.title(title + (" (normalized)" if normalize else ""))
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    thresh = cm_to_plot.max() / 2.0 if cm_to_plot.max() != 0 else 0.5

    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            val = cm_to_plot[i, j]
            text_val = format(val, fmt)  
            plt.text(j, i, text_val,
                     ha="center", va="center",
                     color="white" if val > thresh else "black",
                     fontsize=9)

    plt.tight_layout()
    plt.show()


def show_mistakes(images, y_true, y_pred, max_images=20):
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print("Немає помилок!")
        return

    n = min(max_images, len(wrong_idx))
    chosen = wrong_idx[:n]

    mean, std = 0.1307, 0.3081

    cols = 5
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 2.2, rows * 2.2))

    for k, idx in enumerate(chosen, start=1):
        img = images[idx, 0]  
        img = img * std + mean

        plt.subplot(rows, cols, k)
        plt.imshow(img, interpolation="nearest")
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]}")
        plt.axis("off")

    plt.suptitle("Помилки моделі (True vs Pred)", y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    images, y_true, y_pred = collect_predictions(model, test_loader)

    cm = make_confusion_matrix(y_true, y_pred, num_classes=10)
    plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)], normalize=False, title="MNIST Confusion Matrix")
    plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)], normalize=True, title="MNIST Confusion Matrix")

    show_mistakes(images, y_true, y_pred, max_images=20)