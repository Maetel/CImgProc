## CImgProc
* C++ Image processing algorithms
* Header-only template functions (naive way)
* OpenCL 1.2 implemented image processing algorithms

<img src="/resources/lena example.jpg" width="400px" height="200px" title="px(픽셀) 크기 설정" alt="RubberDuck"></img><br/>

---

### Info
1. Author : Wonjun Hwang
1. E-mail : iamjam4944@gmail.com

---

### Documentation

###### __Build options__
1. `CImgProc` uses `C++ 17` grammar.
1. `CImgProc` is available by including `./include` to your project
1. OpenCL project will be separated in the future

###### __Example dependencies__

Library | Included in `3rdparty`    | Ver. dependency   | Usage                 |
--------|---------------------------|-------------------|-----------------------|
OpenCV  | No                        | 4.1.1 / No        | Image IO              |
Eigen   | Yes                       | 3.3.7 / No        | Matrix computation    |
OpenCL  | Yes                       | 1.2(NVidia) / Yes | Parallel computation  |

*<em>OpenCV version, path to lib files, and path to DLLs must be set manually on `./scripts/link_cv.cmake`</em>

###### __Functions__

* <em>Math</em>
1. Array difference(distance)
1. Welford's online algorithm - 
mean/variance/deviation

* <em>Naive image data manipulation</em>
1. Convert color image to gray
3. Binarization - 
(i) Image histogram 
(ii) Binarization - Threshold/Otsu
4. Template matching - 
(i) Normalized Cross Correlation (NCC) 
(ii) Zero Normalized Cross Correlation (ZNCC)
6. Image convolution -
(i) Convolution with manual kernels 
(ii) Gaussian blur (Gaussian kernel generation) 
(iii) Derivative (naive, Sobel, Scharr) 
(iv) Difference of Gaussian (DoG) 

* <em>OpenCL Implementation</em>
1. Array difference(distance)
1. Convert color image to gray
1. Convolution - 
(i) Convolution with manual kernels 
(ii) Gaussian blur (Gaussian kernel generation) 
(iii) Derivative (naive, Sobel, Scharr) 

---

#### __TODO__
1. separate OpenCL project from example
1. up/down sampling (bilinear, bicubic)
1. demosaic (bayer2rgb)
1. bin packing
1. morphology (erosion & dilation)
