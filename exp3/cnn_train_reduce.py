import copy
import time

import torch
from torch import nn
from torchvision import datasets, transforms


class AlexNet(nn.Module):
    def __init__(self, num_classes=6):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=3),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(32 * 4 * 4, 32),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(32 * 2 * 2, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 2 * 2)
        x = self.classifier(x)
        return x


batch_size = 256

data_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.44087456, 0.39025736, 0.43862119],
                         [0.18242574, 0.19140723, 0.18536106])
])
train_dir = 'Datasets/multiple_animals/train'

train_datasets = datasets.ImageFolder(train_dir, data_transforms)
print("Dataset len:", len(train_datasets))
train_db, val_db = torch.utils.data.random_split(train_datasets, [
    round(len(train_datasets) * 0.8),
    (len(train_datasets) - round(len(train_datasets) * 0.8))
])
image_datasets = {'train': train_db, 'val': val_db}

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x],
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4)
    for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = train_datasets.classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = AlexNet(len(class_names))


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(
                    dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if ('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(
                        dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train_alexnet(net,
                  train_iter,
                  val_iter,
                  optimizer,
                  lr_scheduler,
                  device,
                  num_epochs=50):
    since = time.time()
    net = net.to(device)
    print("Training on", device, "for", num_epochs, "epochs")
    loss = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(net.state_dict())
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time(
        )
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        val_acc = evaluate_accuracy(val_iter, net)
        if val_acc >= best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(net.state_dict())
        print(
            'epoch %d, loss %.4f, train acc %.4f, val acc %.4f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               val_acc, time.time() - start))
        lr_scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    net.load_state_dict(best_model_wts)
    return net


lr, num_epochs = 0.001, 50
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20], gamma=0.1)
net = train_alexnet(net, dataloaders['train'], dataloaders['val'], optimizer,
                    lr_scheduler, device, num_epochs)
torch.save(net, 'cnn_wts_reduce.pth')
