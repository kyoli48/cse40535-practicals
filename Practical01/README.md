# Practical 01: Computer Vision - Color-Based Object Detection

**Course:** CSE 40535 Computer Vision
**University of Notre Dame**

---

## Assignment Overview

This practical implements color-based image processing and video manipulation using OpenCV, including:
- **Task 2:** Multi-object color detection and tracking
- **Task 3:** Invisible cloak effect using color segmentation

---

## Answers to Questions

### Q1: Does HSV work better than RGB for color detection?

**Answer:** Yes, HSV generally works better for color detection because:

- **Hue (H)** represents the color itself and is largely independent of lighting conditions
- **Saturation (S)** represents color intensity/purity
- **Value (V)** represents brightness

In RGB, all three channels (Red, Green, Blue) are affected by lighting changes, making detection less robust. HSV separates chromatic information (H, S) from achromatic information (V), allowing for more reliable color detection across varying lighting conditions.

**Example:** A blue object under bright light and dim light will have very different RGB values, but similar Hue values in HSV.

---

### Q2: Can you ignore one or two channels when working in HSV color space? Why?

**Answer:** Yes! You can often use a wide range for certain channels depending on your detection needs:

- **Ignoring Value (V):** Set a wide V range (e.g., `[50, 255]`) to detect objects under different lighting conditions (bright vs. dim). This works because brightness shouldn't affect which color you're looking for.

- **Ignoring Saturation (S):** Use a wide S range for pastel or faded colors, though be careful as this can also match grayish objects.

- **Hue (H) is critical:** The Hue channel is the most important for color identification and should typically have a narrow, specific range (e.g., `[100, 130]` for blue).

**Best practice:** Keep Hue narrow and specific, use wider ranges for Saturation and Value as needed for robustness.

---

### Q3: What happens with two same-color objects?

**Answer:** The behavior depends on the implementation:

- **Original code (Task 2a):** Only the largest object is detected because the code uses `max(contours, key=cv2.contourArea)`, which selects only the contour with the maximum area.

- **Modified code (Task 2b):** With the modification using a `for` loop to iterate through ALL contours:
  ```python
  for c in contours:
      x, y, w, h = cv2.boundingRect(c)
      # Draw bounding box for each contour
  ```
  Both objects are detected with separate bounding boxes, each labeled individually.

---

## Implementation Details

### Task 2: Color Object Detection

**Objects tracked:**
- Vanilla tootsie roll (cyan/blue): H ~105-110
- Orange tootsie roll: H ~10-15
- Cherry tootsie roll (magenta): H ~165-175

**Key design features:**
- Dictionary-based color configuration for easy scalability
- HSV color space for robust detection
- Morphological operations to reduce noise
- Separate bounding boxes for each detected object

### Task 3: Invisible Cloak

**Cloak used:** Yellow namecard (H ~20-25)

**Algorithm:**
1. Capture static background frame
2. Detect cloak color using HSV thresholding
3. Create mask of cloak region
4. Replace cloak pixels with background pixels using bitwise operations
5. Combine processed frames for invisibility effect

---

## Files Included

- `task2_colorTracking_mac.py` - Multi-color object detection
- `task3_invisibleCloak_mac.py` - Invisible cloak effect
- `colorSelection.py` - HSV histogram analysis tool (provided)
- `README.md` - This file

---

## Usage

### Task 2:
```bash
python task2_colorTracking_mac.py
```
Hold colored objects in front of camera. Press ESC to quit.

### Task 3:
```bash
python task3_invisibleCloak_mac.py
```
1. Step out of frame
2. Press 's' to capture background
3. Step back in with colored cloak
4. Press ESC to quit

---

## HSV Calibration Process

For each object/cloak:
1. Run `colorSelection.py`
2. Press 's' to capture frame
3. Draw ROI around object (tight selection)
4. Analyze HSV histogram peaks
5. Update HSV ranges in code based on peaks

**Tips:**
- Use narrow Hue ranges (±5-10 from peak)
- Use medium Saturation ranges (exclude low saturation to avoid grays)
- Use wide Value ranges for lighting robustness
- High saturation objects (S > 150) work best
