# Theory: Histogram Equalization    

---

## 5. Histogram Equalization

### 5.1 Introduction

Histogram equalization is a global contrast enhancement technique used to improve the visibility of low-contrast images. It works by redistributing pixel intensity values so that the histogram of the output image becomes approximately uniform.

Unlike contrast stretching, which performs a linear transformation, histogram equalization uses a nonlinear transformation based on the cumulative distribution function (CDF).

This technique enhances contrast by spreading frequently occurring intensity values across the entire dynamic range (0–255 for 8-bit images).

---

## 5.2 Probability Distribution Function (PDF)

For a grayscale image with intensity levels ranging from 0 to \( L - 1 \), the probability distribution function (PDF) is defined as:

\[
p_r(r_k) = \frac{n_k}{MN}
\]

Where:

- \( r_k \) = intensity level  
- \( n_k \) = number of pixels with intensity \( r_k \)  
- \( M \times N \) = total number of pixels in the image  
- \( p_r(r_k) \) = probability of occurrence of intensity \( r_k \)

The PDF describes how frequently each gray level appears in the image.

If the histogram is narrow and concentrated, the image has low contrast.

---

## 5.3 Cumulative Distribution Function (CDF)

The cumulative distribution function (CDF) is obtained by accumulating the probabilities from intensity level 0 up to \( r_k \):

\[
CDF(r_k) = \sum_{j=0}^{k} p_r(r_j)
\]

The CDF has the following properties:

- It is monotonically increasing  
- Its values range between 0 and 1  
- It represents cumulative probability  

The CDF is the key component in histogram equalization.

---

## 5.4 Transformation Function

The transformation function used in histogram equalization is:

\[
s_k = (L - 1) \cdot CDF(r_k)
\]

Where:

- \( L \) = total number of intensity levels (256 for 8-bit images)  
- \( s_k \) = new intensity value  
- \( CDF(r_k) \) = cumulative distribution at intensity \( r_k \)

This transformation maps original pixel values to new intensity levels such that the output histogram becomes more uniformly distributed.

---

## 5.5 Why Histogram Equalization Improves Contrast

Histogram equalization enhances contrast by:

- Spreading clustered intensity values over the full dynamic range  
- Increasing differences between neighboring gray levels  
- Making darker regions more distinguishable  

As a result, previously hidden details become more visible, and the overall image appears sharper.

---

## 5.6 Advantages of Histogram Equalization

- Provides strong global contrast enhancement  
- Fully automatic (no parameter tuning required)  
- Effective for dark or low-contrast images  
- Simple to compute  

---

## 5.7 Limitations of Histogram Equalization

- May over-enhance noise  
- Can produce unnatural-looking results  
- Does not preserve overall brightness  
- Not ideal for images with already good contrast  

---

## 5.8 Comparison with Contrast Stretching

| Feature | Contrast Stretching | Histogram Equalization |
|----------|--------------------|------------------------|
| Type of Mapping | Linear | Nonlinear |
| Control | User-defined range | Automatic |
| Contrast Improvement | Moderate | Strong |
| Noise Amplification | Low | Possible |

Histogram equalization performs a nonlinear redistribution of intensity values, which often produces stronger enhancement compared to linear contrast stretching.

---

## 5.9 Summary

Histogram equalization is a powerful image enhancement technique that uses the cumulative distribution function to redistribute intensity values across the full dynamic range. While it significantly improves contrast in low-contrast images, it may also introduce artifacts or amplify noise in certain cases.


