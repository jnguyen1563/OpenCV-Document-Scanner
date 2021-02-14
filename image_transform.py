import numpy as np
import cv2
import argparse
import imutils
from skimage.filters import threshold_local
from transforms import four_point_transform

# Initialize argument parser to receive image
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required=True, help = 'image path')
args =  vars(parser.parse_args())

# read in image
img = cv2.imread(args['image']) # original image

# downscale the image to 500px height to speed up processing
img_downscale = imutils.resize(img, height=500)
# store ratio between original image and downscale
ratio_downscale = img.shape[0] / 500

# process downscaled image and apply edge detection
img_grayscale = cv2.cvtColor(img_downscale, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_grayscale, (5, 5), 0)
img_edge = cv2.Canny(img_blur, 75, 200) # Edge detect image

# find contours from edge detection image
contours = cv2.findContours(img_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
# Get the first 5 largest contours
contours = sorted(contours, key = cv2.contourArea, reverse=True)[:5]

for contour in contours:
    # find an approximation of the contour
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    # if the contour has 4 edges, its probably the document
    if len(approx) == 4:
        document_contour = approx
        break

if 'document_contour' not in locals():
    print('Edges not detected')
    exit()

# draw contours on the downscaled image
img_contour = img_downscale.copy() # downscaled image with contours
cv2.drawContours(img_contour, [document_contour], -1, (0, 255, 0), 2)

# apply 4 point transform to change perspective to top down
img_topdown = four_point_transform(img, document_contour.reshape(4, 2) * ratio_downscale)

# apply grayscale and threshold to look like a scan
img_scanned = cv2.cvtColor(img_topdown, cv2.COLOR_BGR2GRAY)
thresh = threshold_local(img_scanned, 11, offset=10, method='gaussian')
img_scanned = (img_scanned>thresh).astype('uint8') * 255

cv2.imshow('Original', imutils.resize(img, height=650))
cv2.imshow('Edge Detection', imutils.resize(img_edge, height=650))
cv2.imshow('Contour Detection', imutils.resize(img_contour, height=650))
cv2.imshow('Scanned', imutils.resize(img_scanned, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()