import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from random import randint
import io
import base64
from PIL import Image

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
    model.eval()  # Important: evaluation mode

    # Pick a single image from the test set
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

def predict(model, image):
    with torch.no_grad():
        # The model expects a "batch", so we add a dimension with .unsqueeze(0)
        # The image goes from shape (1, 28, 28) to (1, 1, 28, 28)
        output = model(image.unsqueeze(0))
        _, predicted_digit = torch.max(output, dim=1)
    return predicted_digit

def tensor_to_base64(image_tensor):
    """Convert a PyTorch image tensor to a base64-encoded PNG string for the web UI."""
    # Convert tensor (1, 28, 28) to a PIL Image (28, 28)
    image_array = image_tensor.squeeze().numpy()  # Remove channel dim, convert to numpy
    image_array = (image_array * 255).astype('uint8')  # Scale from 0-1 back to 0-255
    pil_image = Image.fromarray(image_array, mode='L')  # 'L' = grayscale
    
    # Resize to 140x140 so it's not tiny in the browser
    pil_image = pil_image.resize((140, 140), Image.NEAREST)
    
    # Save to an in-memory buffer as PNG, then encode as base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    b64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return b64_string