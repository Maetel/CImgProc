//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#include "CImgProc.h"
#include "JamUtility.h"

//std
#include <functional>

//////3rd party
//opencv
#include "opencv2/highgui.hpp"
//eigen
#include "eigen.h"


int main(int argc, char* argv[])
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
				SCOPED_TIMER(histogram gray);
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
	if (0)
	{
		double ZNCCVal = HUGE_VAL;
		{
			SCOPED_TIMER(ZNCC);
			ZNCCVal = ImageAlg::ZNCC(lenaGray.data, lenaDGray.data, pxCount);
		}
		std::cout << "ZNCC value : " << ZNCCVal << std::endl;
	}

#if 0
	//gaussian convolution
	if (1)
	{
		cv::Mat lena_gauss(hi, wid, CV_8U);
		{
			SCOPED_TIMER(gaussian convolution);
			ImageAlg::gaussianConvolotion(lenaGray.data, lena_gauss.data, wid, hi);
		}
		cv::imwrite("lena_gauss.jpg", lena_gauss);
	}
#endif

	std::cout << "Program finished" << std::endl;
	return 0; 
}