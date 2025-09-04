import numpy as np
import matplotlib.pyplot as plt

width = 256
height= 100
filename = "linear3.png"

gradient = np.int32(np.linspace(0, 256, 255)//16)*16
Z = np.tile(gradient, (height, 1)) * 16

plt.figure(figsize=(width / 100, height / 100), dpi=100)
plt.imshow(Z, cmap='gray', origin='upper')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig(filename)
plt.close()
print(f"Image saved as {filename}")
