import imageio.v3 as i
import matplotlib.pyplot as plt
import numpy as np

img = i.imread("C:\\Users\\dhika\\Documents\\pcd\\Zee-JKT48.jpeg",mode ='F')

rx =np.array([
    [1,0],
    [0,-1]
])
ry = np.array([
    [0,1],
    [-1,0]

])

ipad = np.pad(img,pad_width=1,mode='constant', constant_values=0)

gx = np.zeros_like(img)
gy = np.zeros_like(img)

for y in range (1,ipad.shape[0]-1):
    for x in range(1, ipad.shape[1]-1):
        area = ipad[y-1:y+2,x-1:x+2]
        gx[y-1,x-1]=np.sum(area[0:2,0:2]*rx)+ np.sum(area[1:3,1:3]*ry)
        gy[y-1,x-1]=np.sum(area[0:2,0:2]*ry) - np.sum(area[1:3,1:3]*rx)

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