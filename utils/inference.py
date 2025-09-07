import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved_model.h5')
model = tf.keras.models.load_model(model_path)

# Load label encoder
labels_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_labels.csv')
import pandas as pd
labels_df = pd.read_csv(labels_csv)
le = LabelEncoder()
le.fit(labels_df['label'])

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(img_path):
    img = prepare_image(img_path)
    preds = model.predict(img)
    confidence = np.max(preds)
    class_idx = np.argmax(preds)
    class_label = le.inverse_transform([class_idx])[0]
    return class_label, confidence
