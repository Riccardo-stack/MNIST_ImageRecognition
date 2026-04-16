import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from model import MyNetwork

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',           
    train=False,             # Setting False to request the test dataset
    download=True,           
    transform=transform      
)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=64, 
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
)

import torch.nn as nn

model = MyNetwork()
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 5

for epoch in range(NUM_EPOCHS):
    
    # --- TRAINING PHASE ---
    model.train()  # Set the model to "training mode"
    
    total_loss = 0  # Accumulate the loss for this epoch to print it

    for images, labels in train_loader:

        # STEP 1: Forward pass - the network looks at the images and makes its predictions
        predictions = model(images)

        # STEP 2: Calculate how wrong it was
        loss = loss_fn(predictions, labels)

        # STEP 3: Zero out old gradients (ALWAYS before backward!)
        optimizer.zero_grad()

        # STEP 4: Backward pass - PyTorch calculates which direction to adjust the weights
        loss.backward()

        # STEP 5: The optimizer applies the corrections to the weights
        optimizer.step()

        total_loss += loss.item()  # .item() converts the tensor to a regular Python number

    # Print progress at the end of each epoch
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Average loss: {total_loss / len(train_loader):.4f}")

# --- EVALUATION PHASE ---
model.eval()  # Evaluation mode (disables training-only behaviors)

correct = 0
total = 0

with torch.no_grad():  # Tell PyTorch NOT to compute gradients (saves memory)
    
    for images, labels in test_loader:
        
        # The network makes its predictions on the batch
        predictions = model(images)
        
        # predictions has shape (64, 10): for each image, 10 scores
        # torch.max returns the highest value and its INDEX (which is the predicted digit)
        _, predicted_digits = torch.max(predictions, dim=1)
        
        # Compare predicted digits with real ones and count the correct ones
        correct += (predicted_digits == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\nTest Set Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), 'modello_mnist.pth')
print("Model saved to 'modello_mnist.pth'!")
