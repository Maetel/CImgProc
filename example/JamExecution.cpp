//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////
#include "JamExecution.h"

#include "CImgProc.h"
#include "JamUtility.h"

//std
#include <functional>

//////3rd party
//opencv
#include "opencv2/highgui.hpp"
//eigen
#include "eigen.h"
void Jam::execute()
{
	using namespace CIMGPROC;

	cv::samples::addSamplesDataSearchPath(RESOURCES_DIR);
	std::string lenaPath("lena.jpg");					//RGB image
	std::string lenaDoodledPath("lena_doodled.jpg");	//RGB image
	std::string lenaFacePath("lena_face.png");			//RGB image
	auto lenaBGR = cv::imread(cv::samples::findFile(lenaPath));
	auto lenaDoodled = cv::imread(cv::samples::findFile(lenaDoodledPath));
	auto lenaFace = cv::imread(cv::samples::findFile(lenaFacePath));

	const int wid = lenaBGR.cols, hi = lenaBGR.rows, pxCount = wid * hi;
	const int faceWid = lenaFace.cols, faceHi = lenaFace.rows, facePxCount = faceWid * faceHi;

	{
		//debug
		cv::imwrite("lena_copied.jpg", lenaBGR);
	}

	//convert BGR2Gray
	cv::Mat lenaGray(hi, wid, CV_8U);
	{
		SCOPED_TIMER(Lena BGR2Gray);
		ImageAlg::convert2Gray<ImageAlg::RGB2GRAY>(lenaBGR.data, lenaGray.data, pxCount);
	}
	cv::imwrite("lena_gray.jpg", lenaGray);

	cv::Mat lenaDGray(hi, wid, CV_8U);
	{
		SCOPED_TIMER(Lena Doodled BGR2Gray);
		ImageAlg::convert2Gray<ImageAlg::BGR2GRAY>(lenaDoodled.data, lenaDGray.data, pxCount);
	}
	cv::imwrite("lena_doodled_gray.jpg", lenaDGray);

	cv::Mat lenaFaceGray(faceHi, faceWid, CV_8U);
	{
		SCOPED_TIMER(Lena Doodled BGR2Gray);
		ImageAlg::convert2Gray<ImageAlg::BGR2GRAY>(lenaFace.data, lenaFaceGray.data, faceWid * faceHi);
	}
	cv::imwrite("lena_face_gray.jpg", lenaFaceGray);

	//image histogram
	{
		//gray
		if (1)
		{
			MatrixXi histogram_gray(1, 256);
			{
				SCOPED_TIMER(histogram gray);
				ImageAlg::histogram(lenaFaceGray.data, facePxCount, histogram_gray.data());
			}
			//std::cout << histogram_gray << std::endl;
		}
		//BGR
		if (1)
		{
			MatrixXi histogram_BGR(3, 256);
			{
				SCOPED_TIMER(histogram BGR);
				ImageAlg::histogram<3>(lenaFace.data, facePxCount, histogram_BGR.data());
			}
			//std::cout << histogram_BGR << std::endl;
		}
	}

	//NCC
	double NCCVal = HUGE_VAL;
	{
		//of the same size
		if (1)
		{
			{
				SCOPED_TIMER(NCC Val of original and doodled);
				NCCVal = ImageAlg::NCC(lenaGray.data, lenaDGray.data, pxCount);
			}
			std::cout << "NCC Val of original and doodled : " << NCCVal << std::endl;
		}

		// find face
		//  * this naive approach takes tons of time...
		//	* needs to be improved by i)parallel ii)dynamic iii)what else?
		if (0)
		{
			double* outputs = nullptr;
			std::pair<int, int> bestMatchPx({ -1,-1 });	// x, y = col, row
			double bestMatchValue = 0.;
			{
				SCOPED_TIMER(NCC - find face);
				ImageAlg::NCC(
					lenaGray.data, wid, hi,
					lenaFaceGray.data, faceWid, faceHi,
					outputs, &bestMatchPx, &bestMatchValue
				);
			}
			std::cout <<
				"NCC - find face\n" <<
				"Px that best matches (Left top corner) [x,y] = [" << bestMatchPx.first << ", " << bestMatchPx.second << "\n" <<
				"Best matching value : " << bestMatchValue << std::endl;
			{
				delete outputs;
				outputs = nullptr;
			}
		}
	}

	//ZNCC
	if (1)
	{
		double ZNCCVal = HUGE_VAL;
		{
			SCOPED_TIMER(ZNCC);
			ZNCCVal = ImageAlg::ZNCC(lenaGray.data, lenaDGray.data, pxCount);
		}
		std::cout << "ZNCC value : " << ZNCCVal << std::endl;
	}

	//binarization
	cv::Mat lena_otsu(hi, wid, CV_8U);
	if (1)
	{
		{
			SCOPED_TIMER(Otsu binarization);
			ImageAlg::binarize(lenaGray.data, lena_otsu.data, pxCount);
			//int binThres = 50;
			//ImageAlg::binarize<ImageAlg::THRESHOLD>(lenaGray.data, lena_otsu.data, pxCount, &binThres);

		}
		cv::imwrite("lena_Otsu.jpg", lena_otsu);
	}

	//gaussian convolution
	cv::Mat lena_gauss(hi, wid, CV_8U);
	cv::Mat lena_diff(hi, wid, CV_8U);
	if (1)
	{
		{
			SCOPED_TIMER(gaussian convolution);
			ImageAlg::gaussianConvolution(lenaGray.data, lena_gauss.data, wid, hi, 5, 2.0);
		}
		cv::imwrite("lena_gauss.jpg", lena_gauss);

		double diff_sum = 0.;
		{
			SCOPED_TIMER(image difference);
			ImageAlg::distance(lenaGray.data, lena_gauss.data, pxCount, lena_diff.data, &diff_sum);
		}
		std::cout << "Sum of difference : " << diff_sum << std::endl;
		cv::imwrite("lena_diff.jpg", lena_diff);
	}

	//difference of gaussian
	cv::Mat DoG_bin(hi, wid, CV_8U);
	if(1)
	{
		cv::Mat DoG(hi, wid, CV_8U);
		{
			SCOPED_TIMER(difference of gaussian);
			ImageAlg::differenceOfGaussian(lenaGray.data, DoG.data, wid, hi);
		}
		int binThres;
		ImageAlg::binarize(DoG.data, DoG_bin.data, pxCount, &binThres);
		cv::imwrite("lena_DoG_binarized.jpg", DoG_bin);
		std::cout << "Otsu binarization threshold : " << binThres << std::endl;
	}

	//convolution
	cv::Mat lena_sobel(hi, wid, CV_16S);
	if (1)
	{
		std::vector<int> sobel =
		{
			-1, 0, 1,
			-3, 0, 3,
			-1, 0, 1
		};
		{
			SCOPED_TIMER(convolution(sobel));
			ImageAlg::convolution(lena_gauss.data, (short*)lena_sobel.data, wid, hi, sobel.data(), 3, 3);
		}
		cv::imwrite("lena_sobel.jpg", lena_sobel);
	}

	std::cout << "Program finished" << std::endl;
}
