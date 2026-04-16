from flask import Flask, render_template, request, jsonify
from model import load_model, choose_number, predict, tensor_to_base64

app = Flask(__name__)

# Load model and dataset once at startup (not on every request)
print("Loading model and dataset...")
images, test_dataset, model = load_model()
print("Ready!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()
    digit = int(data['digit'])
    
    # Pick a random image of the requested digit
    image, real_label = choose_number(digit, images, test_dataset)
    
    # Run the model
    predicted_digit = predict(model, image)
    predicted = predicted_digit.item()
    
    # Convert image to base64 for the browser
    image_b64 = tensor_to_base64(image)
    
    return jsonify({
        'image_b64': image_b64,
        'real_label': int(real_label),
        'predicted': predicted,
        'correct': predicted == int(real_label)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
