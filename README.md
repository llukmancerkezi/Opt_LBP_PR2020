
**C++ code of the following paper:**

[L. Cerkezi; C. Topal, "Towards more discriminative features for texture recognition", Pattern Recognition, 2020](https://www.sciencedirect.com/science/article/abs/pii/S0031320320302764)


**Warning:** Prior to run the code you need to install opencv 2.4.13.

You can extract many types of LBPs

- Sign - See eq. (1) in the paper
- Mean - See eq. (3) in the paper
- ELBP_A - See eq. (24) in the paper
- SignGlobalTh - It is the same as the Mean but with different threshold value. The threshold is average pixel value of an image.
- MRE_Sign - See eq. (26)
- MRE_Mean - See eq. (27)
- MRE_ELBP_A - See eq. (27)
- MRE_SignGlobalTh - It is the same as the SignGlobalTh but applied the median filter before. See section 4.1.2 in the paper for more detail.

Furthermore you can use three different representations 

- decimal - converting binary code of pixels to decimal value and then obtain the histogram of these values 
- uniform - calculating the uniform value of binary code of pixels and obtain the histogram of these values 
- riu - calculating the riu value of binary code of pixels and then obtain the histogram of these values 

See definitions of *decimal, uniform and riu* in the paper.

You can run the code for three different scenarios:
- Test Case 1. Extracting the final feature vector using only one type of LBP.
- Test Case 2. Using two different LBP types jointly to obtain 2D joint histogram and then obtain the final feature vector either as concatenation of marginal histograms or by flattening the 2D joint histogram.
- Test Case 3. Using two different LBP types jointly and then optimize the 2D joint histogram to obtain more discriminative 2D joint histogram (see the paper for further details). You can obtain the final feature vector either as concatenation of marginal histograms or by flattening the 2D joint optimized histogram.

In the **TestCase.cpp** you can see an example for each case applied on a sample image.

In the **TestCaseClassification.cpp** you can see an example how to make classification on UIUC dataset.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

If you use this code please cite the following paper:

@article{Cerkezi2020PR,
title = {Towards more discriminative features for texture recognition},
author = {Llukman Cerkezi and Cihan Topal},
journal = {Pattern Recognition},
volume = {107},
pages = {107473},
year = {2020},
}
