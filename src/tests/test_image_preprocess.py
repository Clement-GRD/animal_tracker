from predict_test import crop_to_square

# Example test data
image_1 = np.zeros((400, 200, 3))  # Landscape orientation
image_2 = np.zeros((200, 400, 3))  # Portrait orientation
image_3 = np.zeros((300, 300, 3))  # Square image

# Define test cases
@pytest.mark.parametrize("input_image, expected_shape", [
    (image_1, (200, 200, 3)),  # Expected shape for landscape orientation
    (image_2, (200, 200, 3)),  # Expected shape for portrait orientation
    (image_3, (300, 300, 3))   # Expected shape for square image
])
def test_crop_to_square(input_image, expected_shape):
    # Call the function with the input image
    cropped_image = crop_to_square(input_image)
    
    # Check if the cropped image has the expected shape
    assert cropped_image.shape == expected_shape
    
    # Check if the cropped image is indeed square
    assert cropped_image.shape[0] == cropped_image.shape[1]

# Run the tests
if __name__ == "__main__":
    pytest.main()