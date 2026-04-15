import torch.nn as nn
from torch._functorch._aot_autograd.subclass_utils import runtime_unwrap_tensor_subclasses
from networkx import numeric_assortativity_coefficient
from random import randint
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def load_model():
    model = MyNetwork()
    model.load_state_dict(torch.load('modello_mnist.pth'))
    model.eval()  # Importante: modalità valutazione

    # 3. Pick a single image from the test set
    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    
    images = {}
    for index, (image, real_label) in enumerate(test_dataset):
        if real_label not in images:
            images[real_label] = [index]
        else:
            images[real_label].append(index)
    return images, test_dataset, model

def choose_number(x, images, test_dataset):
    if x in images:
        i = randint(0, len(images[x]))
        idx = images[x][i]
        image, real_label = test_dataset[idx]
        return image, real_label
    else:
        return "Invalid number: choose a number from 0 to 9"

def display(image, real_label):
    plt.imshow(image.squeeze(), cmap='gray')  # .squeeze() removes the color channel dimension
    plt.title(f"Real label: {real_label}")
    plt.axis('off')
    plt.show()

def predict(model, image):
    with torch.no_grad():
    # The model expects a "batch", so we add a dimension with .unsqueeze(0)
    # The image goes from shape (1, 28, 28) to (1, 1, 28, 28)
        output = model(image.unsqueeze(0))
        _, predicted_digit = torch.max(output, dim=1)
    return predicted_digit