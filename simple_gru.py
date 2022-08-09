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
sequence_length, input_size = 28, 28  # imagine as we have 28 sequence and each has 28 observations
num_layers = 2  # number of stacked rnn unit
hidden_size = 256  # number of hidden states in each rnn unit
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1


# create a GRU
class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True -> [batch,seq,feature], batch_first=False -> [seq,batch,feature]
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # considering all hidden states
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
        # considering only last hidden state
        # self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.gru(x, h0)
        # to consider all hidden states
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        # to consider only last hidden state
        # out = self.fc(out[:, -1, :])

        return out


# sanity check
model = GRU(28, 256, 2, 10)
x = torch.randn(64, 28, 28)
print(model(x).shape)


# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# initialize network
model = GRU(input_size, hidden_size, num_layers, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device).squeeze(1)  # get to correct shape for GRU: needs (64,28,28) instead (64,1,28,28)
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
            x = x.squeeze(1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')

    model.train()  # evaluation state off, train state on


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
