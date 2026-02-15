# Theory: Low Contrast and Histograms

## Introduction to Contrast in Images

Contrast is the degree of difference between the lightest and darkest parts of an image. Low contrast occurs when this range is narrow, making the image appear dull or "flat".

- **Causes:** Common factors include poor lighting, low-quality sensors, or atmospheric conditions like haze.
- **Impact:** In low-contrast images, details become difficult to distinguish because the intensity values of different objects are too similar.

## Understanding Histograms

An image histogram is a bar chart representing the distribution of pixel intensities.

- **X-Axis:** Represents the intensity levels (typically 0 for black to 255 for white).
- **Y-Axis:** Represents the number of pixels found at each intensity level.

## How Histograms Reveal Contrast Problems

Histograms act as a diagnostic tool for image quality:

- **Narrow Peak:** Indicates low contrast; the pixels are restricted to a small range of values.
- **Bunched on Left:** Indicates an underexposed (dark) image.
- **Bunched on Right:** Indicates an overexposed (bright) image

## Analysis of the Original Histogram

Based on the initial analysis, we can observe the following regarding our test image:

The following metrics were derived from the original grayscale image during the initial analysis phase:

| Metric                 | Value  | Interpretation                                                                                    |
| :--------------------- | :----- | :------------------------------------------------------------------------------------------------ |
| **Min Intensity**      | 10.00  | The image contains very few true blacks, as the lowest value is well above.                       |
| **Max Intensity**      | 199.00 | The image lacks pure whites; the intensity distribution stops significantly before the 255 limit. |
| **Mean Intensity**     | 95.84  | The average brightness (~96) confirms the image is skewed toward the darker side of the spectrum. |
| **Standard Deviation** | 31.34  | A low variance relative to the 0-255 range confirms the low contrast nature of the image.         |

**Conclusion:** These statistics describe an image with a narrow dynamic range. Because the pixels are "clumped" between 10 and 199 rather than spreading across the full 0-255 range, the image appears flat and lacks detail. This data justifies the need for the enhancement techniques implemented later in the project.
