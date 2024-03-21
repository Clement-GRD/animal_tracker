import numpy as np
import cv2 as cv

##################################################################################################################################################################
################################################################ Image Loading and Preprocessing ################################################################
##################################################################################################################################################################

def load_image(path: str) -> np.ndarray:
    """
    Load an image from the specified path and convert it to RGB.

    Parameters:
        path (str): The file path to the image.

    Returns:
        np.ndarray: The loaded image as a NumPy array in RGB format.
    """
    try:
        # Attempt to load the image
        image = cv.imread(path)
        if image is None:
            raise ValueError("Failed to load image from path: {}".format(path))
        
        # Convert from BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return image
    
    except Exception as e:
        raise ValueError("Error loading image: {}".format(e))
    
def read_image_from_buffer(file: Any) -> np.ndarray:
    """
    Reads an image from a buffer and returns it as a NumPy array in RGB format.

    Parameters:
        file (Any): The file buffer containing the image data.

    Returns:
        np.ndarray: The image array in RGB format.
    """
    image = np.frombuffer(file, np.uint8)
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image    

def crop_image_square(image: np.ndarray) -> np.ndarray:
    """
    Crop an input image array into a smaller square image using slice indexing, keeping the center.

    Parameters:
    - image (np.ndarray): The input image array of shape (height, width, channels), where channels represent the color channels.

    Returns:
    - np.ndarray: The cropped square image array of shape (min(height, width), min(height, width), channels).
    """
    # Extracting dimensions of the input image
    height, width, _ = image.shape
    
    # Determine the size of the square crop
    crop_size = min(height, width)
    
    # Calculate the starting indices for the slice to keep the center of the image
    start_y = (height - crop_size) // 2
    start_x = (width - crop_size) // 2
    
    # Perform the crop using slice indexing
    cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size, :]
    
    return cropped_image

def preprocess_image(image, image_size: int = 300):
    """
    Preprocess an image by cropping it into a square, resizing it to the specified size, and expanding its dimension in order to be used as input of an image model.

    Parameters:
        image (np.ndarray): The input image array.
        image_size (int): The desired size of the output image.

    Returns:
        np.ndarray: The preprocessed image array.
    """
    cropped_image = crop_image_square(image)
    resized_image =  cv.resize(cropped_image,(image_size, image_size), interpolation = cv.INTER_AREA)

    return np.expand_dims(resized_image, 0)

if __name__ == '__main__':
    pass