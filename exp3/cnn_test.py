import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

np.set_printoptions(threshold=np.inf)


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


model = torch.load('cnn_wts_reduce.pth')

# save parameters in npy
# params = {}  #change the tpye of 'generator' into dict
# for name, param in model.named_parameters():
#     params[name] = param.detach().cpu().numpy()
# # with open('param_reduce/alexnet_param.txt', 'w') as f:
# for key in params.keys():
#     np.save('param_reduce/' + key + '.npy', params[key])

test_dir = 'Datasets/multiple_animals/test'
test_data = os.listdir(test_dir)

data_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
data_normalize = transforms.Normalize([0.44087456, 0.39025736, 0.43862119],
                                      [0.18242574, 0.19140723, 0.18536106])
# img=Image.open('1.jpg')
# box=img.crop((200,130,440,360))
# re = box.resize((120,120),Image.ANTIALIAS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_label = [
    'batman', 'jienigui', 'miaowazhongzi', 'pikachu', 'saquirrel',
    'xiaohuolong'
]


def test_alexnet(model, _transforms, root_dir, data_list):
    result = {}
    model.eval()
    for data_dir in data_list:
        img = Image.open(os.path.join(root_dir, data_dir))
        inputs = _transforms(img)
        inputs = data_normalize(inputs[0:3]).unsqueeze(0)
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        np.save(data_dir + '.npy', outputs.cpu().detach().numpy())
        result[data_dir] = test_label[preds[0]]
    return result


result = test_alexnet(model, data_transforms, test_dir, test_data)
print(result)
