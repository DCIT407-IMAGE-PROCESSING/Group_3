# Histogram Equalization and Contrast Stretching for Low-Contrast Images

**DCIT 407 — Image Processing Semester Project**  
**Group 3 | Level 400, 1st Semester**

---

## Project Overview

This project explores two fundamental techniques for enhancing low-contrast images:

1. **Contrast Stretching** — A linear transformation that maps the narrow pixel intensity range of a low-contrast image to the full 0–255 range, making details more visible while preserving the original histogram shape.

2. **Histogram Equalization** — A non-linear transformation that uses the Cumulative Distribution Function (CDF) to redistribute pixel intensities, producing a more uniform histogram and stronger contrast enhancement.

We apply both methods to a real low-contrast image, compare the results visually and statistically, and discuss their strengths, limitations, and real-world applications.

## Repository Structure

```
Group_3/
├── DCIT_407_Project_Group.ipynb   # Main project notebook
├── README.md                      # This file
└── Data/
    └── low_contrast_image.jpg     # Test image used in the project
```

## How to Run

1. Open `DCIT_407_Project_Group.ipynb` in **Jupyter Notebook** or **Google Colab**
2. Run all cells from top to bottom
3. The notebook will automatically:
   - Install required libraries (`opencv-python`, `numpy`, `matplotlib`)
   - Download the test image from GitHub
   - Process and display all results

> **Note:** The image is loaded from a URL, so an internet connection is required.

## Technologies Used

- **Python 3**
- **OpenCV** — image loading, grayscale conversion, histogram equalization
- **NumPy** — array operations and contrast stretching computation
- **Matplotlib** — image display and histogram visualization

## Key Results

| Method | Effect |
|---|---|
| Original Image | Narrow intensity range, low contrast |
| Contrast Stretching | Full 0–255 range, natural appearance, histogram shape preserved |
| Histogram Equalization | Uniform histogram, strong contrast boost, may over-enhance |

## References

1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
2. [OpenCV — Histogram Equalization](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
3. [OpenCV — Histograms](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html)
