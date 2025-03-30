import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 훈련 함수
def train(model, train_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# PGD Untargeted 공격 함수
def pgd_untargeted(model, x, label, k, eps, eps_step):
    x_orig = x.clone().detach().to(device)
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True

    for _ in range(k):
        output = model(x_adv)
        loss = F.cross_entropy(output, label.to(device))
        model.zero_grad()
        loss.backward()
        grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv + eps_step * grad_sign  # untargeted는 + 방향
        eta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = x_orig + eta
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()
        x_adv.requires_grad = True

    return x_adv.detach()

# 정확도 측정 함수
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum().item()
            total += len(data)
    return 100 * correct / total

# PGD 공격 정확도 측정
def evaluate_under_attack(model, loader, k, eps, eps_step):
    model.eval()
    correct = 0
    total = 0

    for data, label in loader:
        data, label = data.to(device), label.to(device)
        data_adv = pgd_untargeted(model, data, label, k, eps, eps_step)
        output = model(data_adv)
        pred = output.argmax(dim=1)
        correct += pred.eq(label).sum().item()
        total += len(data)

    return 100 * correct / total

# 메인 실행
if __name__ == "__main__":
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = SimpleCNN().to(device)
    train(model, train_loader, epochs=3)

    clean_acc = evaluate(model, test_loader)
    print(f"\n[Clean Accuracy] {clean_acc:.2f}%")

    # PGD 공격 파라미터
    k = 10
    eps = 0.3
    eps_step = 0.03

    adv_acc = evaluate_under_attack(model, test_loader, k, eps, eps_step)
    print(f"[PGD Untargeted Attack Accuracy] eps={eps}, k={k} → {adv_acc:.2f}%")
