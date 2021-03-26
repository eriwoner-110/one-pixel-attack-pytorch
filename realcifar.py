import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                          shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                         shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),-1)


# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Define a convolutional neural network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define a VGG16 neural network

class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.flatten = Flatten()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),


            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # nn.Dropout(0.2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),


            nn.MaxPool2d(2, 2),
            # nn.Dropout(0.2),

            self.flatten,
            nn.Dropout(),
            nn.Linear(512,2048),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(2048),
            # nn.Dropout(0.2),
            nn.Dropout(),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(2048),
            #nn.Dropout(0.2),
            nn.Linear(2048,10),
        )

    def forward(self, x):
        return self.model(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.flatten = Flatten()
        self.model = nn.Sequential(nn.Conv2d(3,64,3,padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64,128,3,padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d((2,2)),
                                   nn.Conv2d(128,256,3,padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d((2,2)),
                                   self.flatten,
                                   nn.Linear(256*8*8,1024),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.2),
                                   nn.Linear(1024,256),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.2),
                                   nn.Linear(256,10),
                                   )

    def forward(self, x):
        x = self.model(x)
        return x


def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    return np.mean(np.equal(predictions.cpu().numpy(), y.cpu().numpy()))


def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, acccuracy"""
    # Gather losses
    losses = []
    batch_accuracies = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in tqdm(data):
        # Grab x,y
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Get output prediction
        outputs = model(inputs)

        # Predict and store accuracy
        predictions = torch.argmax(outputs, dim=1)
        batch_accuracies.append(compute_accuracy(predictions, labels))

        # Compute losses
        loss = F.cross_entropy(outputs, labels)
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy


def train_model(train_data, dev_data, model, lr=0.01, momentum=0.9, nesterov=False, n_epochs=100):
    """Train a model for N epochs given data and hyper-params."""
    # We optimize with SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, n_epochs + 1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Run **training***
        loss, acc = run_epoch(train_data, model.train(), optimizer)
        print('Train | loss: {:.6f}  accuracy: {:.6f}'.format(loss, acc))
        losses.append(loss)
        accuracies.append(acc)

        # Run **validation**
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer)
        print('Valid | loss: {:.6f}  accuracy: {:.6f}'.format(val_loss, val_acc))
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save model
        path = './cifar1_net.pth'
        torch.save(model.state_dict(), path)

    return losses,accuracies,val_losses,val_accuracies


if __name__ == '__main__':

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(images,labels)
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # Configure the device to train on
    use_cude = True
    print('Cuda Available:', torch.cuda.is_available())
    device = torch.device('cuda:2' if (use_cude and torch.cuda.is_available()) else 'cpu')

    # # Display size of the data
    # print(train_set.data.shape)
    # print(test_set.data.shape)

    model = VGG16().to(device)

    train_model(train_loader,test_loader,model)