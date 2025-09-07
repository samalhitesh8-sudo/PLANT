import numpy as np
from tensorflow.keras.preprocessing import image

def prepare_image(img_path, target_size=(128, 128)):
    """
    Load and preprocess an image for model inference.

    Args:
        img_path (str): Path to the image file.
        target_size (tuple): The desired image size (width, height).

    Returns:
        np.ndarray: Preprocessed image ready for prediction.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array
