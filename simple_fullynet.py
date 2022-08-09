import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5
load_model = False


def load_checkpoint(checkpoint):
    print('Checkpoint loaded.')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('Checkpoint saved.')
    torch.save(state, filename)


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
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')
    model.train()  # evaluation state off, train state on


# create fullynet
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# sanity check
model_X = NN(784, 10)  # number of features, num_classes
x = torch.randn(64, 784)  # num of observation, number of features
print(f'shape model_X: {model_X(x).shape}')

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
print(f'Training started with {num_epochs} epoch(s)')

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))

for epoch in range(num_epochs):

    losses = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # get to correct shape (flatten an image (64,1,28,28) to (64,284))
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_loss = sum(losses)/len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:.5f}')

    # checkpoint
    if epoch % 2 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

# accuracy check
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
