# MNIST Image Classifier

A handwritten digit classifier built from scratch with **PyTorch**, trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) (60,000 training images, 10,000 test images).

## Architecture

A simple feed-forward neural network with three fully connected layers:

```
Input (784) → Linear (128) → ReLU → Linear (64) → ReLU → Linear (10) → Output
```

Achieves **~97.5% accuracy** on the test set after 5 epochs of training.

## Project Structure

| File | Description |
|------|-------------|
| `model.py` | Network architecture (`MyNetwork`) and utility functions |
| `main.py` | Training pipeline: downloads data, trains the model, evaluates and saves weights |
| `predict.py` | Loads saved weights and classifies individual images from the test set |

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision matplotlib
```

## Usage

**Train the model:**
```bash
python main.py
```
This downloads the MNIST dataset, trains the network for 5 epochs, prints the test accuracy, and saves the weights to `modello_mnist.pth`.

**Predict a digit:**
```bash
python predict.py
```
Edit the number inside `choose_number()` in `predict.py` (0–9) to pick a digit category, and the model will classify a random image of that digit.

## Requirements

- Python 3.10+
- PyTorch
- Torchvision
- Matplotlib
