import numpy as np
from PIL import Image, ImageOps

def load_preprocess_image(image_path: str, image_size: int = 300) -> np.ndarray:
    """
    Load and preprocess an image from the specified file path.

    Parameters:
        image_path (str): The file path to the image.
        image_size (int): The desired size of the cropped image. Defaults to 300.

    Returns:
        np.ndarray: The preprocessed image as a NumPy array.
    """
    try:
        # Open the image, convert it to RGB, and crop it to the specified size
        image = Image.open(image_path).convert('RGB')
        cropped_image = ImageOps.fit(image, (image_size, image_size))
        image_array = np.expand_dims(np.array(cropped_image), 0)
        return image_array
    
    except Exception as e:
        # If an error occurs, raise an exception with an error message
        raise ValueError(f"Error loading or preprocessing image: {str(e)}")