from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from preprocessing.affact_dataset import AffactDataset
from network.resnet_51 import resnet50

plt.ion()


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                black = outputs[j][8]
                blonde = outputs[j][9]
                brown = outputs[j][10]
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('Black: {:.2f}  blonde: {:.2f} brown: {:.2f}'.format(black, blonde, brown))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 6}


dataset_train = AffactDataset(transform=data_transforms, max_size=1000, index_offset=0)
dataset_val = AffactDataset(transform=data_transforms, max_size=200, index_offset=1000)

training_generator = torch.utils.data.DataLoader(dataset_train, **params)
validation_generator = torch.utils.data.DataLoader(dataset_val, **params)

dataloaders = {
    'train': training_generator,
    'val': validation_generator
}

dataset_sizes = {
    'train': len(dataset_train),
    'val': len(dataset_val)
}

model = resnet50(pretrained=True)
model_ft = model.to(device)
# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.00001, momentum=0.9)
checkpoint = torch.load('latest_model.pt')
model_ft.load_state_dict(checkpoint['model_state_dict'])
optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
visualize_model(model_ft, 6)