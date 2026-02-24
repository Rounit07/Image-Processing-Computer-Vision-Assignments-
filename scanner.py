# Name:
# Roll No:
# Course: Image Processing & Computer Vision
# Unit:
# Assignment Title: Smart Document Scanner & Quality Analysis System
# Date:

print("Welcome to Smart Document Scanner & Quality Analysis System")
print("This system analyzes sampling and quantization effects on document quality.")

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("document.jpg")
image = cv2.resize(image, (512, 512))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.show()
high = cv2.resize(gray, (512, 512))
medium = cv2.resize(gray, (256, 256))
low = cv2.resize(gray, (128, 128))
medium_up = cv2.resize(medium, (512, 512))
low_up = cv2.resize(low, (512, 512))
quant_4bit = np.floor(gray / 16) * 16
quant_2bit = np.floor(gray / 64) * 64
plt.figure(figsize=(12,8))

plt.subplot(2,4,1)
plt.imshow(gray, cmap='gray')
plt.title("Original")

plt.subplot(2,4,2)
plt.imshow(high, cmap='gray')
plt.title("512x512")

plt.subplot(2,4,3)
plt.imshow(medium_up, cmap='gray')
plt.title("256x256")

plt.subplot(2,4,4)
plt.imshow(low_up, cmap='gray')
plt.title("128x128")

plt.subplot(2,4,5)
plt.imshow(gray, cmap='gray')
plt.title("8-bit")

plt.subplot(2,4,6)
plt.imshow(quant_4bit, cmap='gray')
plt.title("4-bit")

plt.subplot(2,4,7)
plt.imshow(quant_2bit, cmap='gray')
plt.title("2-bit")

plt.tight_layout()
plt.show()

print("Observations:")
print("1. Lower resolution reduces text clarity and edge sharpness.")
print("2. Lower bit depth reduces smoothness and detail.")
print("3. High resolution and 8-bit depth are best suited for OCR.")