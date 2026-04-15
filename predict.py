from model import load_model, choose_number, display, predict

# 2. Load the empty structure and insert saved weights
images, test_dataset, model = load_model()

# Change the number below to ask the model to "classify" the image
image, real_label = choose_number(0, images, test_dataset)

# 4. Display the image with matplotlib
display(image, real_label)

# 5. Make the prediction
predicted_digit = predict(model, image)

print(f"Network prediction:  {predicted_digit.item()}")
if real_label == predicted_digit.item():
    print("Correct prediction!")
else:
    print("Wrong prediction!")