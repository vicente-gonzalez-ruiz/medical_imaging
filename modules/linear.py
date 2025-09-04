import numpy as np
import matplotlib.pyplot as plt

width = 256
height= 100
filename = "linear.png"

gradient = np.linspace(0, 255, width)
Z = np.tile(gradient, (height, 1))

plt.figure(figsize=(width / 100, height / 100), dpi=100)
plt.imshow(Z, cmap='gray', origin='upper')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig(filename)
plt.close()
print(f"Image saved as {filename}")
