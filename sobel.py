import imageio.v3 as i
import matplotlib.pyplot as plt
import numpy as np

img = i.imread("c:\\Users\\Lenovo\\Downloads\\gemes.jpeg",mode ='F')

sx =np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
])
sy = np.array([
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]

])

ipad = np.pad(img,pad_width=1,mode='constant', constant_values=0)

gx = np.zeros_like(img)
gy = np.zeros_like(img)

for y in range (1,ipad.shape[0]-1):
    for x in range(1, ipad.shape[1]-1):
        area = ipad[y-1:y+2,x-1:x+2]
        gx[y-1,x-1]=np.sum(area*sx)
        gy[y-1,x-1]=np.sum(area*sy)

G = np.sqrt(gx**2 +gy**2)

G=(G/G.max())*255
G= np.clip(G,0,255)
G =G.astype(np.uint8)
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.imshow(img)

plt.subplot(2,2,2)
plt.imshow(gx,cmap='gray')

plt.subplot(2,2,3)
plt.imshow(gy,cmap='gray')

plt.subplot(2,2,4)
plt.imshow(G,cmap='gray')

plt.show()