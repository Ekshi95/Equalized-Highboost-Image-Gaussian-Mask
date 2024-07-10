# -*- coding: utf-8 -*-


import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


path = "C:/Users/Talip/Desktop/goruntu3.png"
img = cv2.imread(path,0)


equ = cv2.equalizeHist(img)

h,c = equ.shape

scale_up_x = 2
scale_up_y = 2

scaled_f_up = cv2.resize(equ, None, fx= scale_up_x, fy= scale_up_y, interpolation= cv2.INTER_CUBIC)




gauss = cv2.GaussianBlur(scaled_f_up, (7,7), 0)

unsharp_image = cv2.addWeighted(scaled_f_up, 2, gauss, -0.5, 0)




row, col = unsharp_image.shape

I = cv2.dft(np.float32(unsharp_image),flags = cv2.DFT_COMPLEX_OUTPUT)
I_shift = np.fft.fftshift(I)

magnitude_spectrum = 20*np.log(cv2.magnitude(I_shift[:,:,0],I_shift[:,:,1]))

D0=200


H=[[math.exp(-((i-col/2)**2+(j-row/2)**2)/(2*D0**2)) for i in range(col)] for j in range(row)]


If=np.zeros((row,col,2))


If[:,:,0]=I_shift[:,:,0]*H
If[:,:,1]=I_shift[:,:,1]*H

magnitude_spectrum1 = 20*np.log(cv2.magnitude(If[:,:,0],If[:,:,1]))
f_ishift = np.fft.ifftshift(If)
img_back = cv2.idft(f_ishift)



plt.rcParams["figure.figsize"] = [15, 7]
plt.rcParams["figure.autolayout"] = True

hist,bins = np.histogram(img.flatten(),256,[0,256])
plt.subplot(6, 2, 2).hist(img.flatten(),256,[0,256], color = 'b')
plt.title('Orginal Histogram')


hist,bins = np.histogram(equ.flatten(),256,[0,256])
plt.subplot(6, 2, 4).hist(equ.flatten(),256,[0,256], color = 'b')
plt.title("Equalized Histogram")

hist,bins = np.histogram(scaled_f_up.flatten(),256,[0,256])
plt.subplot(6, 2, 6).hist(scaled_f_up.flatten(),256,[0,256], color = 'b')
plt.title("Scaled Equalized Histogram")

hist,bins = np.histogram(unsharp_image.flatten(),256,[0,256])
plt.subplot(6, 2, 8).hist(unsharp_image.flatten(),256,[0,256], color = 'b')
plt.title("Highboost Histogram")

plt.subplot(6, 2, 1)
plt.title("Orginal Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(6, 2, 3)
plt.title("Equalized Image")
plt.imshow(cv2.cvtColor(equ, cv2.COLOR_BGR2RGB))

plt.subplot(6, 2, 5)
plt.title("Scaled Equalized Image")
plt.imshow(cv2.cvtColor(scaled_f_up, cv2.COLOR_BGR2RGB))

plt.subplot(6, 2, 7)
plt.title("Highboost Image")
plt.imshow(cv2.cvtColor(unsharp_image, cv2.COLOR_BGR2RGB))

plt.subplot(6, 2, 9)
plt.title("Highboost Image")
plt.imshow(cv2.cvtColor(unsharp_image, cv2.COLOR_BGR2RGB))

plt.subplot(6,2,10)
plt.imshow(magnitude_spectrum, cmap = 'gray',vmin = 0, vmax = 255)
plt.title('Highboost Magnitude Spectrum')

plt.subplot(6,2,11)
plt.imshow(H, cmap = 'gray')
plt.title('Gaussian Mask')

plt.subplot(6,2,12)
plt.imshow(img_back[:,:,0], cmap = 'gray')
plt.title('Gaussian Filtered Image')

plt.show()


