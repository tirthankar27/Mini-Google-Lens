#Extract the .tar file
import tarfile

tar_path = "./dataset/256_ObjectCategories.tar"
extract_path = "./dataset"

with tarfile.open(tar_path) as tar:
    tar.extractall(path=extract_path)

print("Extraction done")
#Check dataset structure
import os

base_path = "./dataset/256_ObjectCategories"
classes = sorted(os.listdir(base_path))
classes = [c for c in classes if os.path.isdir(os.path.join(base_path, c))]

print("Number of classes:", len(classes))
print("Sample classes:", os.listdir(base_path)[:5])
#Image count
class_counts = {}
total_images = 0

for cls in classes:
    cls_path = os.path.join(base_path, cls)
    images = [img for img in os.listdir(cls_path) if img.endswith(".jpg")]
    
    count = len(images)
    class_counts[cls] = count
    total_images += count

print("Total Images:", total_images)
#Sample images
import matplotlib.pyplot as plt
import random

sample_images = []

for cls in random.sample(classes, 9):
    cls_path = os.path.join(base_path, cls)
    img_name = random.choice(os.listdir(cls_path))
    sample_images.append(os.path.join(cls_path, img_name))

fig, axes = plt.subplots(3, 3, figsize=(8,8))

for ax, img_path in zip(axes.flatten(), sample_images):
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_title(img_path.split("/")[-2], fontsize=8)
    ax.axis("off")

plt.show()
#Class Distribution
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

print("Top 5 Classes:")
for cls, count in sorted_classes[:5]:
    print(cls, count)

print("\nBottom 5 Classes:")
for cls, count in sorted_classes[-5:]:
    print(cls, count)
#Image dimensions
widths = []
heights = []

import random

sample_paths = []

for cls in random.sample(classes, 20):
    cls_path = os.path.join(base_path, cls)
    img_name = random.choice(os.listdir(cls_path))
    sample_paths.append(os.path.join(cls_path, img_name))

for path in sample_paths:
    img = plt.imread(path)
    heights.append(img.shape[0])
    widths.append(img.shape[1])

plt.hist(widths, bins=20)
plt.title("Image Width Distribution")
plt.show()

plt.hist(heights, bins=20)
plt.title("Image Height Distribution")
plt.show()
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root="./dataset/256_ObjectCategories")

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_indices, test_indices = random_split(range(len(full_dataset)), [train_size, test_size])

train_data = Subset(
    datasets.ImageFolder(root="./dataset/256_ObjectCategories", transform=train_transform),
    train_indices.indices
)

test_data = Subset(
    datasets.ImageFolder(root="./dataset/256_ObjectCategories", transform=test_transform),
    test_indices.indices
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=32, num_workers=0)

print("Train size:", len(train_data))
print("Test size:", len(test_data))

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
#Load pretained model (transfer learning)
import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = models.resnet50(pretrained=True)

# Modify final layer
num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

print("Model ready with", num_classes, "classes")
#Training setup
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
#Training loop
epochs = 8

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
#Test accuracy
all_preds = []
all_labels = []

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Accuracy:", 100 * correct / total, "%")
#Mini Google Lense
from PIL import Image
import matplotlib.pyplot as plt

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    class_name = full_dataset.classes[predicted.item()]

    plt.imshow(img)
    plt.title(f"Prediction: {class_name}")
    plt.axis("off")
    plt.show()


predict_image("frog.jpg")
torch.save(model.state_dict(), "model.pth")
import json

with open("classes.json", "w") as f:
    json.dump(full_dataset.classes, f)

torch.save(model.state_dict(), "model.pth")

print("Model and classes saved!")