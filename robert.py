import imageio.v3 as i
import matplotlib.pyplot as plt
import numpy as np

img = i.imread("c:\\Users\\Lenovo\\Downloads\\gemes.jpeg", mode='F')

robert_x = np.array([
    [1, 0],
    [0, -1]
])
robert_y = np.array([
    [0, 1],
    [-1, 0]
])

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

ipad = np.pad(img, pad_width=1, mode='constant', constant_values=0)

# Inisialisasi hasil deteksi tepi
robert_gx = np.zeros_like(img)
robert_gy = np.zeros_like(img)
sobel_gx = np.zeros_like(img)
sobel_gy = np.zeros_like(img)

for y in range(1, ipad.shape[0]-1):
    for x in range(1, ipad.shape[1]-1):
        area = ipad[y-1:y+2, x-1:x+2]
        robert_gx[y-1, x-1] = np.sum(area[:2, :2] * robert_x)
        robert_gy[y-1, x-1] = np.sum(area[:2, :2] * robert_y)

for y in range(1, ipad.shape[0]-1):
    for x in range(1, ipad.shape[1]-1):
        area = ipad[y-1:y+2, x-1:x+2]
        sobel_gx[y-1, x-1] = np.sum(area * sobel_x)
        sobel_gy[y-1, x-1] = np.sum(area * sobel_y)

robert_g = np.sqrt(robert_gx**2 + robert_gy**2)
sobel_g = np.sqrt(sobel_gx**2 + sobel_gy**2)

robert_g = (robert_g / robert_g.max()) * 255
sobel_g = (sobel_g / sobel_g.max()) * 255

robert_g = np.clip(robert_g, 0, 255)
sobel_g = np.clip(sobel_g, 0, 255)

robert_g = robert_g.astype(np.uint8)
sobel_g = sobel_g.astype(np.uint8)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(img)

plt.subplot(2, 2, 2)
plt.imshow(robert_g, cmap='gray')

plt.subplot(2, 2, 3)
plt.imshow(sobel_g, cmap='gray')

plt.subplot(2, 2, 4)
plt.imshow(img)

plt.show()