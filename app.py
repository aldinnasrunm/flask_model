import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import keras_cv

model = tf.keras.models.load_model('yolo_model.keras')

def preprocess_single_image(file_storage):
    try:
        # Read the file as bytes
        img_bytes = file_storage.read()
        print("File read successfully.")
        
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        print("Image decoded successfully.")
        
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, (640, 640))
        img = tf.expand_dims(img, axis=0)
        print("Image preprocessed successfully.")
        
        return img
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({'error': 'No file provided'})

        try:
            data = preprocess_single_image(file)           
            y_pred = model.predict(data, verbose=0)
            y_pred_serializable = {
                'boxes': y_pred['boxes'].tolist(),
                'confidence': y_pred['confidence'].tolist(),
                'classes': y_pred['classes'].tolist(),
                'num_detections': y_pred['num_detections'].tolist()
            }

            np_array = np.array(y_pred_serializable['boxes'])
            valid_boxes = np.sum(~np.all(np_array == [-1.0, -1.0, -1.0, -1.0], axis=2))
            return jsonify({"Number of valid boxes" : int(valid_boxes)}) 
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': str(e)})

    return "OK"

if __name__ == '__main__':
    app.run(debug=True)

