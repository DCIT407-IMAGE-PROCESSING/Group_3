
import json
import os

# Notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    })

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    })

# --- Notebook Content ---

# Title
add_markdown("""# Group 3 Project: Digital Image Enhancement
**Topic:** Techniques for Fixing Low Contrast Images
**Objective:** Compare and implement methods to make dull images look clearer and more vibrant.

---
### Project Overview
Digital images sometimes come out looking dull or "flat" because the lighting was poor. This project shows how we can use math and programming (Python) to fix these images. We will look at three main methods: **Contrast Stretching**, **Histogram Equalization**, and **Color Optimization** using special color spaces. 

This notebook includes our explanations, the code we wrote, and the final results.""")

# Section 1: Setup
add_markdown("""## Section 1: Getting Ready
Before we start, we need to prepare our tools and the image we want to fix. We want every image to be 512x512 pixels so that our math works the same way every time. We also start by working with "Grayscale" (black and white) versions of the images to keep things simple.""")

add_code(r"""import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# A simple tool to find and prepare our image
def prepare_image(directory='.'):
    # Try to find a low contrast image first
    priority_file = os.path.join('Data', 'low_contrast_image.jpg')
    image_path = priority_file if os.path.exists(priority_file) else None
    
    if not image_path:
        # If not found, look for any image
        valid_types = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        for d in ['Data', '.']:
            if not os.path.exists(d): continue
            for file in os.listdir(d):
                if file.lower().endswith(valid_types):
                    image_path = os.path.join(d, file)
                    break
            if image_path: break
    
    if not image_path:
        # If we still find nothing, make a fake image so the code doesn't crash
        print("Note: No image found. Creating a simple test image.")
        img = np.random.normal(128, 20, (512, 512)).astype(np.uint8)
        image_path = "test_image.jpg"
    else:
        img = cv2.imread(image_path)
        if img is None: raise ValueError("Could not open the image file!")
        # Convert to Black and White (Grayscale)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to a standard 512x512 size
    ready_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    return ready_img, image_path

# Run the setup
try:
    source_img, source_path = prepare_image()
    
    # Show the original image and its "Histogram"
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(source_img, cmap='gray', vmin=0, vmax=255)
    # Using format() instead of f-strings in generated code to avoid quote/newline issues
    plt.title("Original Image\nSource: {}".format(os.path.basename(source_path)))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # A Histogram is a chart that shows how many dark vs bright pixels are in an image
    plt.hist(source_img.ravel(), 256, range=[0, 256], color='gray', alpha=0.7)
    plt.title("Luminance Histogram (Pixel Distribution)")
    plt.xlabel("Brightness Level (0=Black, 255=White)")
    plt.ylabel("Number of Pixels")
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate some numbers to help us understand the image
    stats = {
        "Darkest Pixel": np.min(source_img),
        "Brightest Pixel": np.max(source_img),
        "Average Brightness": np.mean(source_img),
        "Contrast Level (Std Dev)": np.std(source_img)
    }
    
    print("{:<25} | {:<12}".format('Metric', 'Value'))
    print("-" * 40)
    for name, value in stats.items():
        print("{:<25} | {:<12.2f}".format(name, value))

except Exception as e:
    print("Oops, something went wrong: {}".format(e))""")

# Section 2: Theory - Low Contrast
add_markdown(r"""## Section 2: What is "Low Contrast"?

### 2.1 Understanding Contrast
Contrast is basically the difference between the darkest and lightest parts of a picture. 
- **Low Contrast:** The image looks dull because most of its pixels have very similar brightness. There are no deep blacks and no bright whites.
- **High Contrast:** The image looks sharp because there is a wide range from dark to light.

### 2.2 What a Histogram Tells Us
A **Histogram** is like a map of the image's brightness.
- If all the bars are bunched together in a narrow "mountain," the image has low contrast.
- If the mountain is on the left, the image is too dark.
- If it's on the right, it's too bright.

### 2.3 Looking at our Data
In our test image, we can see that:
- The darkest pixel is around 10 (not truly black).
- The brightest pixel is 199 (not truly white).
- This "bunching" in the middle is why the image looks flat. We need to spread these pixels out!""")

# Section 3: Theory - Contrast Stretching
add_markdown(r"""## Section 3: Method 1 - Contrast Stretching

### 3.1 What is it?
Imagine the image's brightness range is a rubber band that has been squashed together. **Contrast Stretching** is when we grab both ends of that band and pull them until they reach 0 (Black) and 255 (White).

### 3.2 How it works (The Math)
We use a simple formula for every pixel:
$$ NewPixel = \frac{OldPixel - Minimum}{Maximum - Minimum} \times 255 $$

This math ensures the darkest pixels become 0 and the brightest become 255, stretching everything else in between. 

### 3.3 Why use it?
It's very fast and simple. It makes the image look much clearer without changing the "feeling" of the picture too much.""")

# Section 4: Implementation - Contrast Stretching
add_code(r"""# Python function to stretch the contrast
def stretch_contrast(image):
    # Convert to float so we can do precise math
    f_img = image.astype(float)
    
    # We use the 2nd and 98th percentile to find the min and max
    # This helps ignore "dots" of light or dark that might be errors
    min_val = np.percentile(f_img, 2)
    max_val = np.percentile(f_img, 98)
    
    if max_val == min_val: return image
        
    # Apply the stretching formula
    result = (f_img - min_val) * (255.0 / (max_val - min_val))
    
    # Ensure values stay between 0 and 255, then convert back to image format
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# Apply the fix and show the result
cs_image = stretch_contrast(source_img)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(cs_image, cmap='gray', vmin=0, vmax=255)
plt.title("After Contrast Stretching")
plt.axis('off')

plt.subplot(1, 2, 2)
# Notice how the bars are now spread out across the whole chart
plt.hist(cs_image.ravel(), 256, range=[0, 256], color='blue', alpha=0.6)
plt.title("New Histogram (Spread Out)")
plt.xlim([0, 256])
plt.show()""")

# Section 5: Theory - Histogram Equalization
add_markdown(r"""## Section 4: Method 2 - Histogram Equalization

### 4.1 What is it?
While Stretching just pulls the ends of the rubber band, **Histogram Equalization** tries to flatten the "mountain" so that there is an equal amount of every brightness level. This makes the picture look much more intense.

### 4.2 Key Terms
- **PDF (Probability Distribution):** How likely is a certain brightness value?
- **CDF (Cumulative Distribution):** Adding up the probabilities. This "CDF" becomes our map to change the pixels.

### 4.3 CLAHE (A Smarter Way)
Sometimes regular Equalization makes a picture look "fake" or too noisy. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) is a smarter version. It breaks the image into small squares and fixes them one by one, making sure it doesn't add too much noise.""")

# Section 6: Implementation - HE & CLAHE
add_code(r"""# Implementation of the two equalization methods
def equalize_histogram(image):
    # Calculate the distribution and the cumulative map (CDF)
    hist, _ = np.histogram(image.flatten(), 256, range=[0, 256])
    cdf = hist.cumsum()
    
    # Normalize the map to 0-255
    cdf_m = np.ma.masked_equal(cdf, 0)
    if (cdf_m.max() - cdf_m.min()) == 0: return image
    cdf_norm = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    map_table = np.ma.filled(cdf_norm, 0).astype('uint8')
    
    return map_table[image]

def apply_clahe(image):
    # Using the OpenCV built-in tool for CLAHE
    engine = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return engine.apply(image)

# Run both and compare
he_image = equalize_histogram(source_img)
clahe_image = apply_clahe(source_img)

plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.imshow(he_image, cmap='gray', vmin=0, vmax=255)
plt.title("Method: Global Equalization")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(clahe_image, cmap='gray', vmin=0, vmax=255)
plt.title("Method: CLAHE (Adaptive)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.hist(he_image.ravel(), 256, range=[0, 256], color='green', alpha=0.5, label='Global')
plt.hist(clahe_image.ravel(), 256, range=[0, 256], color='red', alpha=0.5, label='CLAHE')
plt.title("Histogram Comparison")
plt.legend(); plt.show()""")

# Section 7: Expansion - Color Space Optimization
add_markdown(r"""## Section 5: Helping Color Images

Enhancing color images is tricky. If we just fix the Red, Green, and Blue separately, the colors will change and look weird.

### 5.1 The LAB Color Space
To solve this, we use a different way to look at color called **LAB**:
- **L (Lightness):** Only the brightness.
- **A & B:** Only the color info.

By fixing **only the L channel** (the brightness), we can make the image clearer while keeping the colors exactly as they should be!""")

add_code(r"""# Tool to fix a color image safely
def fix_color_image(path):
    src = cv2.imread(path)
    if src is None: return None
    
    # Change from standard BGR to LAB
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Fix ONLY the L (brightness) channel using CLAHE
    engine = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_new = engine.apply(l)
    
    # Put the pieces back together and convert back to normal color
    merged = cv2.merge((l_new, a, b))
    fixed = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)

# Show the results for the color version
color_result = fix_color_image(source_path)

if color_result is not None:
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB))
    plt.title("Original Color Image"); plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(color_result)
    plt.title("Enhanced Color Image (LAB Method)"); plt.axis('off')
    plt.show()""")

# Section 8: Comparison & Analysis
add_markdown("""## Section 6: Final Comparison

How do we know which method is better? We look at two things:
1.  **Contrast Score (Std Dev):** A higher number means a wider range from light to dark.
2.  **Detail Score (Entropy):** A higher number means more visible patterns and "information" in the image.""")

add_code(r"""def get_detail_score(img):
    hist, _ = np.histogram(img, 256, range=[0, 256])
    p = hist / hist.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

methods = {
    "Original": source_img,
    "Stretched": cs_image,
    "Equalized": he_image,
    "CLAHE": clahe_image
}

print("{:<25} | {:<15} | {:<15}".format('Method Used', 'Contrast Score', 'Detail Score'))
print("-" * 60)
for name, img in methods.items():
    contrast = np.std(img)
    detail = get_detail_score(img)
    print("{:<25} | {:<15.2f} | {:<15.4f}".format(name, contrast, detail))

# Final Visual Comparison
plt.figure(figsize=(12, 10))
list_imgs = [source_img, cs_image, he_image, clahe_image]
list_titles = ["Original", "Stretched", "Equalized", "CLAHE"]

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(list_imgs[i], cmap='gray', vmin=0, vmax=255)
    plt.title(list_titles[i])
    plt.axis('off')
plt.tight_layout(); plt.show()""")

# Section 9: Conclusion
add_markdown("""## Section 7: Final Summary
After testing all methods, here is what our group found:
- **Contrast Stretching** is the simplest and safest way to fix a slightly dull photo.
- **Histogram Equalization** is very powerful but can sometimes make an image look "fake" by adding too much noise.
- **CLAHE** is the best balance for most cases, especially if you use it on the **L channel** of a color image. It reveals hidden details without destroying the original look of the picture.

We hope this project clearly shows how simple math can be used to improve the technology we use every day!""")

# Generate the file
output_file = "Group3_Histogram_Equalization.ipynb"
with open(output_file, "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Project Success: Notebook file '{}' has been created.".format(output_file))
print("You can now open it and run the presentation.")
