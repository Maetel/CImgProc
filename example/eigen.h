//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////
#ifdef CIMG_LINK_EIGEN

#ifndef CIMGPROC_EIGEN_H
#define CIMGPROC_EIGEN_H
#include <Eigen/Dense>

using namespace Eigen;

using VectorRGB = Matrix<unsigned char, 3, 1>;
using MatrixXf  =  Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXuc =  Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic>;
#endif //!CIMGPROC_EIGEN_H

#endif //!CIMG_LINK_EIGEN