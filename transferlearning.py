import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epochs = 1


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# load pre-trained network and freeze its parameters
# (only modified layers after the freezing will be updated)
model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# modify the model based on the data we have
print(model)  # want to remove avgpool and also change the classifier
model.avgpool = Identity()

# 1.Changing the whole classifier
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 10))
model.to(device)

# 2.Changing a part of the classifier (don't forget to make rest of the classifier Identity)
# model.classifier[0] = nn.Linear(512,10)
# model.to(device)

# load data
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# accuracy check
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data...')
    else:
        print('Checking accuracy on test data...')
    num_correct = 0
    num_samples = 0
    model.eval()  # evaluation state on, train state off

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')

    model.train()  # evaluation state off, train state on


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
