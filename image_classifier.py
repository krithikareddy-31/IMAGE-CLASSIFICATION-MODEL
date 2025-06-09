import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("Starting CIFAR-10 Image Classification...")

# Set random seed and default dtype for reproducibility
torch.manual_seed(42)
torch.set_default_dtype(torch.float32)  # Ensure we use float32 for better compatibility

# 1. Load and Preprocess the Dataset
print("Loading and preprocessing the CIFAR-10 dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                        shuffle=True, num_workers=0)

# Download and load the test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                       shuffle=False, num_workers=0)

# Print dataset information
print(f"Training dataset size: {len(trainset)}")
print(f"Test dataset size: {len(testset)}")

# 2. Define a simple CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the model
device = "cpu"  # Force CPU usage for simplicity
print(f"Using device: {device}")
model = Net().to(device)

# 3. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 4. Train the Model
print("Training the model...")
num_epochs = 5  # Reduced number of epochs for faster training

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# 5. Test the model
print("Testing the model...")
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test images: {100 * correct / total:.2f}%')

"""Loading and preprocessing the CIFAR-10 dataset...
Test dataset size: 10000
Using device: cpu
Training the model...
[Epoch 1, Batch 100] Loss: 2.160
[Epoch 1, Batch 200] Loss: 1.861
[Epoch 1, Batch 300] Loss: 1.642
[Epoch 1, Batch 400] Loss: 1.533
[Epoch 1, Batch 500] Loss: 1.448
[Epoch 1, Batch 600] Loss: 1.401
[Epoch 1, Batch 700] Loss: 1.347
[Epoch 2, Batch 100] Loss: 1.239
[Epoch 2, Batch 200] Loss: 1.198
[Epoch 2, Batch 300] Loss: 1.171
[Epoch 2, Batch 400] Loss: 1.135
[Epoch 2, Batch 500] Loss: 1.114
[Epoch 2, Batch 600] Loss: 1.091
[Epoch 2, Batch 700] Loss: 1.059
[Epoch 3, Batch 100] Loss: 0.967
[Epoch 3, Batch 200] Loss: 0.949
[Epoch 3, Batch 300] Loss: 0.934
[Epoch 3, Batch 400] Loss: 0.948
[Epoch 3, Batch 500] Loss: 0.923
[Epoch 3, Batch 600] Loss: 0.911
[Epoch 3, Batch 700] Loss: 0.896
[Epoch 4, Batch 100] Loss: 0.776
[Epoch 4, Batch 200] Loss: 0.786
[Epoch 4, Batch 300] Loss: 0.780
[Epoch 4, Batch 400] Loss: 0.808
[Epoch 4, Batch 500] Loss: 0.765
[Epoch 4, Batch 600] Loss: 0.778
[Epoch 4, Batch 700] Loss: 0.768
[Epoch 5, Batch 100] Loss: 0.633
[Epoch 5, Batch 200] Loss: 0.632
[Epoch 5, Batch 300] Loss: 0.633
[Epoch 5, Batch 400] Loss: 0.676
[Epoch 5, Batch 500] Loss: 0.661
[Epoch 5, Batch 600] Loss: 0.645
[Epoch 5, Batch 700] Loss: 0.635
Finished Training
Testing the model...
Accuracy on test images: 70.03% """