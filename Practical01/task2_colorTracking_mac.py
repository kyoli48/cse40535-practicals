# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2024
# ________________________________________________________________
# Adam Czajka, Andrey Kuehlkamp, September 2017 - 2024

# Here are your tasks:
#
# Task 2a:
# - Select one object that you want to track and set the RGB
#   channels to the selected ranges (found by colorSelection.py).
# - Check if HSV color space works better. Can you ignore one or two
#   channels when working in HSV color space? Why?
# - Try to track candies of different colors (blue, yellow, green).
#
# Task 2b:
# - Adapt your code to track multiple objects of *the same* color simultaneously,
#   and show them as separate objects in the camera stream.
#
# Task 2c:
# - Adapt your code to track multiple objects of *different* colors simultaneously,
#   and show them as separate objects in the camera stream. Make your code elegant
#   and requiring minimum changes when the number of different objects to be detected increases.
#
# Task for students attending 60000-level course:
# - Choose another color space (e.g., LAB or YCrCb), modify colorSelection.py, select color ranges
#   and after some experimentation say which color space was best (RGB, HSV or the additional one you selected).
#   Try to explain the reasons why the selected color space performed best.

# =============================================================================
# ANSWERS TO QUESTIONS:
#
# Q: Does HSV work better than RGB?
# A: Yes, HSV generally works better for color detection because:
#    - Hue (H) represents the color itself and is largely independent of lighting
#    - Saturation (S) represents color intensity
#    - Value (V) represents brightness
#    In RGB, all three channels are affected by lighting changes, making detection
#    less robust. HSV separates chromatic (H,S) from achromatic (V) information.
#
# Q: Can you ignore one or two channels when working in HSV color space? Why?
# A: Yes! You can often use a wide range for Value (V) to ignore brightness
#    variations. For example, setting V range to [50, 255] allows detection
#    under different lighting conditions. Similarly, you can use a wide
#    Saturation range for pastel colors. The Hue channel is the most important
#    for color identification and should have a narrow, specific range.
#
# Q: What happens with two same-color objects?
# A: Without modification (original code), only the largest object is detected
#    because the code uses max(contours, key=cv2.contourArea). With Task 2b
#    modifications (using a for loop), both objects are detected with separate
#    bounding boxes.
# =============================================================================

import cv2
import numpy as np

# =============================================================================
# TASK 2c: Color configuration dictionary for multiple different colors
# To add a new color, simply add a new entry to this dictionary.
# Use colorSelection.py to find the appropriate HSV ranges for your objects.
# =============================================================================
COLORS = {
    'VANILLA': {
        # Vanilla tootsie roll: H ~105-110, S ~40-80, V ~120-150
        'lower': np.array([100, 30, 100]),
        'upper': np.array([115, 100, 170]),
        'bgr': (255, 200, 100),              # BGR color for bounding box (Light blue)
        'label': 'VANILLA'
    },
    'ORANGE': {
        # Orange tootsie roll: H ~10-15, S ~180-220 (HIGH), V ~180-220
        'lower': np.array([8, 160, 160]),
        'upper': np.array([18, 240, 240]),
        'bgr': (0, 165, 255),                # BGR color for bounding box (Orange)
        'label': 'ORANGE'
    },
    'CHERRY': {
        # Cherry tootsie roll: H ~165-175, S ~200-240 (HIGH), V ~180-220
        'lower': np.array([160, 180, 160]),
        'upper': np.array([178, 255, 240]),
        'bgr': (255, 0, 255),                # BGR color for bounding box (Magenta)
        'label': 'CHERRY'
    }
}

cam = cv2.VideoCapture(0)

# Ignore bounding boxes smaller than "minObjectSize"
minObjectSize = 60  # Increased to filter out small false detections

while (True):
    retval, img = cam.read()

    res_scale = 0.5  # rescale the input image if it's too large
    img = cv2.resize(img, (0, 0), fx=res_scale, fy=res_scale)

    # =============================================================================
    # TASK 2a & 2c: Convert to HSV color space (more robust to lighting changes)
    # =============================================================================
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Morphological operations kernel (larger to merge nearby regions)
    kernel = np.ones((7, 7), np.uint8)

    # =============================================================================
    # TASK 2c: Loop through all colors in our configuration dictionary
    # This design makes it easy to add/remove colors - just modify the COLORS dict
    # =============================================================================
    for color_name, color_config in COLORS.items():

        # Create mask for current color using HSV ranges
        objmask = cv2.inRange(hsv, color_config['lower'], color_config['upper'])

        # Apply aggressive morphological operations to merge nearby regions
        objmask = cv2.morphologyEx(objmask, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
        objmask = cv2.morphologyEx(objmask, cv2.MORPH_DILATE, kernel=kernel, iterations=2)

        # Find connected components
        cc = cv2.connectedComponents(objmask)
        ccimg = cc[1].astype(np.uint8)

        # Find contours of these objects
        contours, hierarchy = cv2.findContours(ccimg,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)[-2:]

        # =============================================================================
        # TASK 2b: Loop through ALL contours (not just the largest one)
        # This allows tracking multiple objects of the same color simultaneously
        # =============================================================================
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # Do not show very small objects
            if w > minObjectSize or h > minObjectSize:
                # Draw bounding box with color-specific BGR value
                cv2.rectangle(img, (x, y), (x + w, y + h), color_config['bgr'], 3)

                # Add label with color name
                cv2.putText(img,
                            color_config['label'],      # text (color name)
                            (x, y - 10),                # start position
                            cv2.FONT_HERSHEY_SIMPLEX,   # font
                            0.7,                        # size
                            color_config['bgr'],        # BGR color (matches box)
                            2,                          # thickness
                            cv2.LINE_AA)                # type of line

    cv2.imshow("Live WebCam", img)

    action = cv2.waitKey(1)
    if action == 27:
        break

cam.release()
cv2.destroyAllWindows()