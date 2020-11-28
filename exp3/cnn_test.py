import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

np.set_printoptions(threshold=np.inf)


class RCNet(nn.Module):
    def __init__(self, num_classes=6):
        super(RCNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 3 * 3, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 3 * 3)
        x = self.classifier(x)
        return x


def test_rcnet(model, _transforms, root_dir, data_list, data_normalize,
               test_label, device):
    result = {}
    model.eval()
    for data_dir in data_list:
        img = Image.open(os.path.join(root_dir, data_dir))
        inputs = _transforms(img)
        inputs = data_normalize(inputs[0:3]).unsqueeze(0)
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        result[data_dir] = test_label[preds[0]]
    return result


def main():
    # load model
    model = torch.load('cnn_wts_reduce.pth')

    # save parameters in npy
    params = {}
    for name, param in model.named_parameters():
        params[name] = param.detach().cpu().numpy()

    for key in params.keys():
        np.save('param/' + key + '.npy', params[key])

    test_dir = 'Datasets/multiple_animals/test'
    test_data = os.listdir(test_dir)

    data_transforms = transforms.Compose([
        transforms.CenterCrop(300),
        transforms.Resize((24, 24)),
        transforms.ToTensor()
    ])
    data_normalize = transforms.Normalize([0.44087456, 0.39025736, 0.43862119],
                                          [0.18242574, 0.19140723, 0.18536106])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_label = [
        'batman', 'jienigui', 'miaowazhongzi', 'pikachu', 'saquirrel',
        'xiaohuolong'
    ]

    result = test_rcnet(model, data_transforms, test_dir, test_data,
                        data_normalize, test_label, device)
    print(result)


if __name__ == '__main__':
    main()
