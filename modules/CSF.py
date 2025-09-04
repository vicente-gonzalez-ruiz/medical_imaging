import numpy as np
import matplotlib.pyplot as plt
#from skimage.draw import checkerboard

def generate_csf_image_with_gratings(output_filename="csf_plot_with_gratings.png"):
    """
    Generates and saves a PNG image of the Contrast Sensitivity Function (CSF)
    along with a visual representation of grayscale gratings matching the CSF shape.

    Args:
        output_filename (str, optional): The name of the output PNG file. 
                                         Defaults to "csf_plot_with_gratings.png".
    """

    # --- CSF Plot Generation (same as before) ---
    f = np.logspace(np.log10(0.1), np.log10(45), 1000) # Adjusted max frequency for visual match
    
    # Parameters for the CSF model (log-parabolic function)
    gamma_max = 500  # peak sensitivity
    f_max = 4.0      # peak spatial frequency
    beta = 2.0       # bandwidth
    delta = 0.05     # truncation value

    kappa = np.log10(2)
    beta_prime = np.log10(2 * beta)

    s_prime = np.log10(gamma_max) - kappa * \
        (np.log10(f) - np.log10(f_max) * beta_prime / 2)**2

    s = np.where(
        (f < f_max) & (s_prime < np.log10(gamma_max) - delta),
        np.log10(gamma_max) - delta,
        s_prime
    )
    sensitivity = 10**s
    contrast = 1 / sensitivity # Contrast is inverse of sensitivity

    # Normalize contrast to be between 0 and 1 for visual representation
    contrast = contrast / np.max(contrast) 
    
    # --- Grayscale Grating Generation ---
    img_height = 200 # Height of the grating image
    img_width = 800  # Width of the grating image (corresponds to spatial frequency range)
    gratings_image = np.zeros((img_height, img_width))

    # We need to map the spatial frequencies to the image width
    # This will be a non-linear mapping (logarithmic)
    # Let's create an array of spatial frequencies for the image columns
    sf_image_cols = np.logspace(np.log10(0.1), np.log10(45), img_width)

    for i in range(img_width):
        sf_at_col = sf_image_cols[i]
        
        # Find the corresponding contrast from the CSF curve for this spatial frequency
        # We need to interpolate the contrast values
        current_contrast_from_csf = np.interp(sf_at_col, f, contrast)

        # Generate a sine grating with the specific spatial frequency and contrast
        # The number of cycles across a short segment of the image to get the frequency
        # A full cycle is 1 unit of spatial frequency. To map it to pixels,
        # consider the spatial frequency as cycles per degree, and map degrees to pixels.
        # Here, we'll simplify and just make the number of cycles scale with sf_at_col.
        
        # Let's assume img_height corresponds to 1 degree for simplicity in grating generation
        # This will make sf_at_col directly relate to cycles within img_height pixels.
        
        # The number of cycles that fit into one 'unit' (e.g., 1 degree) needs to be mapped to the pixel space.
        # If we want 'sf_at_col' cycles per degree, and our image effectively represents 'X' degrees,
        # then total cycles for that column segment would be sf_at_col * X.
        # Let's just generate a vertical sine wave pattern for each column strip
        x = np.linspace(0, sf_at_col * 2 * np.pi, img_height) # Scales the number of cycles
        grating = 0.5 + 0.5 * current_contrast_from_csf * np.sin(x)
        gratings_image[:, i] = grating

    # --- Plotting ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 3]}) # Adjust for two plots

    # Plot 1: Grayscale Gratings Image
    ax[0].imshow(gratings_image, cmap='gray', aspect='auto', origin='lower',
                   extent=[np.log10(0.1), np.log10(45), 0, 1]) # Use log scale for extent

    ax[0].set_ylabel("Contraste")
    ax[0].set_xticks([]) # Hide x-axis ticks for the image
    ax[0].set_yticks([0, 1])
    ax[0].set_yticklabels(['0', '1'])


    # Plot 2: CSF Curve
    ax[1].plot(f, contrast, color='black', linewidth=2, label="CSF Contrast Threshold")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log') # Keep y-scale log for contrast threshold
    ax[1].set_xlabel("Spatial Frequency [cpd]")
    ax[1].set_ylabel("Contrast Threshold")
    ax[1].grid(True, which="both", ls="--", alpha=0.7)
    
    # Overlay the CSF curve on the gratings image for direct comparison
    # Need to map the log-scaled spatial frequency on the x-axis for both plots
    # We will use the same x-axis for both plots, scaled logarithmically
    ax[0].plot(np.log10(f), contrast, color='black', linewidth=2)


    # Adjust x-axis ticks to show actual spatial frequencies, not log values
    tick_locs = np.array([0.1, 1, 4, 10, 45])
    ax[1].set_xticks(tick_locs)
    ax[1].set_xticklabels([str(x) for x in tick_locs])

    # Make the top plot (gratings) transparent to show the curve on it
    ax[0].set_facecolor((0.8, 0.8, 0.8)) # A light gray background
    
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"CSF plot with gratings saved as {output_filename}")

if __name__ == '__main__':
    generate_csf_image_with_gratings()
