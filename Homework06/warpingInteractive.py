# Special Studies: Computer Vision (CSE 40535)
# University of Notre Dame
# ______________________________________________________________________
# Adam Czajka, Toan Q. Nguyen, Siamul Khan, Walter Scheirer 2016 -- 2025

import numpy as np
import math
import cv2
from ROIPoly import roiPoly

roi = roiPoly(sort=False)

# Get selected image from roi poly object
I = cv2.cvtColor(np.array(roi.origImage), cv2.COLOR_RGB2BGR)
rows, cols, channels = I.shape

# Get transformation matrix using points from roi
src = np.float32(roi.points)
dst = np.float32([[0, 0], [cols-1, 0], [cols-1, rows-1], [0, rows-1]])
# dst = np.float32([[cols-1, 0], [0, 0], [0, rows-1], [cols-1, rows-1]]) # dst matrix if sort is enabled

H_mat = cv2.getPerspectiveTransform(src, dst)

# Inverse of H maps destination pixel coords back to source pixel coords
H_inv = np.linalg.inv(H_mat)

# Having matrix H we may do our transformation for each pixel:
I_transformed = np.zeros(I.shape).astype(np.uint8)

count = 0
for y_dest in range(0, rows):
    for x_dest in range(0, cols):

        destPX = np.float32([[x_dest], [y_dest], [1]])

        # *** MODIFIED LINE 1: apply H_inv to find source coords from destination
        # (original was: destPX = H_mat @ sourcePX — forward warp, causes holes)
        sourcePX = H_inv @ destPX

        x_source = int(sourcePX[0,0]/sourcePX[2,0])
        y_source = int(sourcePX[1,0]/sourcePX[2,0])

        if x_source > 0 and y_source > 0 and x_source < cols and y_source < rows:
            count = count + 1

            # *** MODIFIED LINE 2: read from source, write to destination
            I_transformed[y_dest, x_dest, :] = I[y_source, x_source, :]

I_correct_xformed = cv2.warpPerspective(I, H_mat, (cols, rows), flags=cv2.INTER_NEAREST)

cv2.imshow('Warped Images (left is yours, right is the correct one from library implementation)', np.concatenate([I_transformed, I_correct_xformed], axis=1))
print('This version of warping calculated new values for ', 100*count/(rows*cols), '% of destination pixels.')
cv2.waitKey(0)
cv2.destroyAllWindows()