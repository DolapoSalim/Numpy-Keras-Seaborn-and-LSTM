import cv2
import numpy as np
import matplotlib.pyplot as plt

#using opencv to read an image
#BGR Image
image = cv2.imread("C:/users/public/pictures/underexposed_image.jpg")

#seperating colour channels
B = image[:,:,0] #blue layer
G = image[:,:,1] #green layer
R = image[:,:,2] #red layer

#equilize each channel seperately
b_equi = cv2.equalizeHist(B)
g_equi = cv2.equalizeHist(G)
r_equi = cv2.equalizeHist(R)

#calculate histograms for each channel seperately
B_histo = cv2.calcHist([b_equi],[0], None, [256], [0,256]) 
G_histo = cv2.calcHist([g_equi],[0], None, [256], [0,256])
R_histo = cv2.calcHist([r_equi],[0], None, [256], [0,256])

#merge thechannels and create new image
equi_im = cv2.merge([b_equi,g_equi,r_equi])

#visualize the equilized channels seperately
plt.imshow(b_equi)
plt.title("b_equi")
plt.show()
plt.imshow(g_equi)
plt.title("g_equi")
plt.show()
plt.imshow(r_equi)
plt.title("r_equi")
plt.show()

#visualize the channel histograms seperately
plt.subplot(2, 2, 1)
plt.plot(G_histo, 'g')
plt.subplot(2, 2, 2)
plt.plot(R_histo, 'r')
plt.subplot(2, 2, 3)
plt.plot(B_histo, 'b')

#visualize the original and equilized images
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL);
cv2.imshow("Original Image",image);
cv2.namedWindow("New Image", cv2.WINDOW_NORMAL);
cv2.imshow("New Image",equi_im);

cv2.waitKey(0) & 0xFF 
cv2.destroyAllWindows()
