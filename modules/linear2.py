import numpy as np
import matplotlib.pyplot as plt

width = 256
height= 100
filename = "linear2.png"

x = np.linspace(0, width, width)
y = np.linspace(0, height, height)
X, Y = np.meshgrid(x, y)
Z = X*Y

plt.figure(figsize=(width / 100, height / 100), dpi=100)
plt.imshow(Z, cmap='gray', origin='upper')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig(filename)
plt.close()
print(f"Image saved as {filename}")
