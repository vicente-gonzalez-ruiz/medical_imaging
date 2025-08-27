import numpy as np
from PIL import Image

# This script requires the Pillow library. You can install it with:
# pip install Pillow

def generate_8bit_grayscale_image(width, height, output_filename):
    """
    Generates an 8-bit grayscale PNG image with a linear horizontal gradient.

    Parameters:
    - width (int): The width of the image in pixels.
    - height (int): The height of the image in pixels.
    - output_filename (str): The name of the output PNG file.

    The 8-bit mode in Pillow is 'L'. Pixel values range from 0 (black) to 255 (white).
    """
    print(f"Generating 8-bit grayscale image: {output_filename}")
    
    # Create a NumPy array with a horizontal gradient.
    # The data type 'uint8' is used for 8-bit images.
    data = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        # Calculate the pixel value based on the x-coordinate.
        # This creates a gradient from 0 to 255.
        data[:, x] = int(x / width * 255)

    # Create a new Pillow image from the NumPy array in 'L' mode.
    # 'L' stands for Luminance, which is the standard 8-bit grayscale mode.
    image = Image.fromarray(data, 'L')
    
    # Save the image as a PNG file.
    image.save(output_filename, format='PNG')
    print(f"Successfully created {output_filename}")

def generate_16bit_grayscale_image(width, height, output_filename):
    """
    Generates a 16-bit grayscale PNG image with a linear horizontal gradient.

    Parameters:
    - width (int): The width of the image in pixels.
    - height (int): The height of the image in pixels.
    - output_filename (str): The name of the output PNG file.

    The 16-bit mode in Pillow is 'I;16'. Pixel values range from 0 to 65535.
    """
    print(f"Generating 16-bit grayscale image: {output_filename}")

    # Create a NumPy array with a horizontal gradient.
    # The data type 'uint16' is used for 16-bit images.
    data = np.zeros((height, width), dtype=np.uint16)
    for x in range(width):
        # Calculate the pixel value based on the x-coordinate.
        # This creates a gradient from 0 to 65535.
        data[:, x] = int(x / width * 65535)

    # Create a new Pillow image from the NumPy array in 'I;16' mode.
    # 'I;16' is the 16-bit integer grayscale mode.
    image = Image.fromarray(data, 'I;16')

    # Save the image as a PNG file.
    image.save(output_filename, format='PNG')
    print(f"Successfully created {output_filename}")

if __name__ == "__main__":
    # Define the dimensions for the images.
    img_width = 1024
    img_height = 128
    
    # Generate and save the 8-bit image.
    generate_8bit_grayscale_image(img_width, img_height, "grayscale_8bit.png")
    
    # Generate and save the 16-bit image.
    generate_16bit_grayscale_image(img_width, img_height, "grayscale_16bit.png")
