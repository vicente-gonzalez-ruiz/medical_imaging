import numpy as np
import matplotlib.pyplot as plt

def generate_cosine_art(width=800, height=600, frequency_scaling=0.01, filename='CSF3.png'):
    """
    Generates and saves an image of a cosine function with variable frequency and amplitude.

    Args:
        width (int): The width of the image in pixels.
        height (int): The height of the image in pixels.
        frequency_scaling (float): A factor to control the frequency change.
        filename (str): The name of the file to save the image to.
    """
    # Create a grid of (X, Y) coordinates
    x = np.linspace(0, width/2, width)
    y = np.linspace(height-1, 0, height)
    #x = np.log(x+1)
    #y = np.exp(1/(x+1))
    y = 1/(y/50+1)
    #y = np.log(y/10000+1)
    X, Y = np.meshgrid(x, y)

    # Calculate the Z values based on the cosine function
    # Amplitude is proportional to Y
    # Frequency changes with X (phase is proportional to X^2)
    Z = Y * (1-np.sin(frequency_scaling * X**3))

    # Create the plot
    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    
    # Use imshow to display the 2D array as an image.
    # 'gray' colormap gives a nice black and white effect.
    plt.imshow(Z, cmap='gray', origin='upper')

    # Remove axes and ticks for a clean image
    plt.axis('off')
    
    # Adjust layout to prevent any borders
    plt.tight_layout(pad=0)

    # Save the figure
    plt.savefig(filename)
    plt.close()
    print(f"Image saved as {filename}")

if __name__ == '__main__':
    # You can change the parameters here to get different results
    generate_cosine_art(width=1920, height=1080, frequency_scaling=0.000001)
