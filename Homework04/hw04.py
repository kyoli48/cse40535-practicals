import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load our pattern
gray = cv2.imread("pattern.png",cv2.IMREAD_GRAYSCALE)

############################################################################
#    Step 1: build the Gabor kernel that will enhance for us vertically oriented patches:
#    cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
#
#    where:
#    ksize  - size of kernel in pixels (n, n), i.e., size of our neighborhood
#    sigma  - size of the Gaussian envelope, i.e., how wide is our Gaussian "hat"
#    theta  - orientation of the normal to the filter's oscilation pattern; e.g., theta = 0.0 means vertical stripes
#    lambda - wavelength of the sinusoidal oscilation; this together with sigma 
#             is resposinble for frequencies enhanced by this filter
#    gamma  - spatial aspect ratio; keep it 1
#    phi    - phase offset; keep it 0
#    ktype  - type and range of values that each pixel in the gabor kernel can hold; keep it cv2.CV_32F

# ***TASK*** Select parameters of your Gabor kernel here:
ksize = 7       # try something between 5 and 15
sigma = 4.0     # try something between 2.0 and 4.0
theta = 0.0     # keep it 0.0 if you want to focus on vertically-oriented patterns
lbd = 4.0       # try something between 2.0 and 4.0
gamma = 1.0     # keep it 1.0
psi = 0.0       # keep it 0.0

kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lbd, gamma, psi, ktype=cv2.CV_32F)

# Normalize the kernel and remove the DC component (do you remember from our class discussion why we are doing this?)
kernel /= kernel.sum()
kernel -= kernel.mean()

# Curious how does the kernel look like? Here we go:
xx, yy = np.mgrid[0:kernel.shape[0], 0:kernel.shape[1]]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xx, yy, kernel ,rstride=1, cstride=1, cmap=plt.cm.gray,linewidth=0)
plt.savefig("gabor_kernel_visualization.png")
plt.close()
print("Gabor kernel visualization saved to gabor_kernel_visualization.png")


############################################################################
# Step 2: image filtering

res1 = cv2.filter2D(gray, cv2.CV_8UC3, kernel)


############################################################################
# Step 3: image binarization (let's use an idea with maximization of the Fisher ratio, implemeted by Otsu)

th2, res2 = cv2.threshold(res1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


############################################################################
# Step 4: morphological operations and getting your area of interest annotated
# ***TASK*** Choose the type among cv2.MORPH_CLOSE, cv2.MORPH_OPEN, cv2.MORPH_ERODE or cv2.MORPH_DILATE
# (or a sequence of those, in the order you think makes sense)

se_small = np.ones((3,3), np.uint8)
se_med = np.ones((9,9), np.uint8)

# MORPH_CLOSE with small kernel to locally connect nearby white pixels
res3 = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel=se_small, iterations=3)
# MORPH_OPEN with tiny kernel to gently remove sparse noise
res3 = cv2.morphologyEx(res3, cv2.MORPH_OPEN, kernel=se_small, iterations=1)
# MORPH_CLOSE to solidify blocks
res3 = cv2.morphologyEx(res3, cv2.MORPH_CLOSE, kernel=se_med, iterations=2)
# MORPH_ERODE to tighten boundaries and sharpen edges
res3 = cv2.morphologyEx(res3, cv2.MORPH_ERODE, kernel=se_small, iterations=3)

############################################################################
# Step 5 (Bonus): Edge detection to demarcate vertical pattern regions
# Use Canny edge detection on the morphological result, then overlay on original image

# Find contours of the morphological mask and draw white edges on black background
contours, _ = cv2.findContours(res3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
res4 = np.zeros_like(gray)
cv2.drawContours(res4, contours, -1, 255, 2)

# Save the results for documentation
cv2.imwrite("result_step1_filtering.png", res1)
cv2.imwrite("result_step2_binarization.png", res2)
cv2.imwrite("result_step3_morphological.png", res3)
cv2.imwrite("result_step4_edges.png", res4)

print("\nResults saved:")
print("  - result_step1_filtering.png")
print("  - result_step2_binarization.png")
print("  - result_step3_morphological.png")
print("  - result_step4_edges.png (bonus - edge detection)")

# Display windows if GUI is available (comment out for headless execution)
# Uncomment the following lines to view results interactively:
# cv2.imshow("Filtering result",res1)
# cv2.imshow("Otsu's binarization",res2)
# cv2.imshow("Areas with vertical pattern annotated",res3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("\nDone!")