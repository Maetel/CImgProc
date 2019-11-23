## CImgProc
C++ Image processing algorithms
Header-only template functions

<img src="/resources/lena example.jpg" width="400px" height="200px" title="px(픽셀) 크기 설정" alt="RubberDuck"></img><br/>

---

### Info
Author : Wonjun Hwang
E-mail : iamjam4944@gmail.com
Contributor : Brett Ahn

---

### Documentation

###### __Build options__
`CImgProc` uses C++ 17 grammar.
`CImgProc` is available by including `./include` to your project

###### __Example dependencies__

Library | Included in `3rdparty`    | Ver. dependency   | Usage                 |
--------|---------------------------|-------------------|-----------------------|
OpenCV  | No                        | 4.1.1 / No        | Image IO              |
Eigen   | Yes                       | 3.3.7 / No        | Matrix computation    |
OpenCL  | Yes                       | 1.2(NVidia) / Yes | Parallel computation  |
*<em>OpenCV version, path, and DLLs must be set manually on `CMakeLists.txt`</em>

###### __Functions__

* <em>Math</em>
    1. Array difference(distance)
    1. Welford's online algorithm
    mean/variance/deviation

* <em>Image</em>
    1. Convert color image to gray
    RGB/RGBA to gray
    BGR/BGRA to gray
    3. Binarization
    Image histogram
    Binarization - Threshold/Otsu
    4. Template matching
    Normalized Cross Correlation (NCC)
    Zero Normalized Cross Correlation (ZNCC)
    6. Image convolution
    Convolution with manual kernels
    Gaussian blur (Gaussian kernel generation)
    Derivative (naive, Sobel, Scharr)
    Difference of Gaussian (DoG)

---

#### __TODO__
    1. up/down sampling (bilinear, bicubic)
    1. demosaic (bayer2rgb)
    1. bin packing
    1. morphology (erosion & dilation)