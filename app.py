from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained YOLO model
model = YOLO('best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file temporarily
    file_path = 'temp.jpg'
    file.save(file_path)

    # Perform garbage detection
    results = model.predict(source=file_path, save=True, save_txt=True)

    # Extract detection results
    output = []
    for result in results:
        for box in result.boxes:
            output.append({
                'class': model.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy.tolist()[0]
            })

    # Clean up the temporary file
    os.remove(file_path)

    return jsonify({'predictions': output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)