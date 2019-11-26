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
			int histogram_gray[256];
			{
				SCOPED_TIMER(histogram gray);
				ImageAlg::histogram(lenaFaceGray.data, facePxCount, histogram_gray);
			}
			//std::cout << histogram_gray << std::endl;
		}
		//BGR
		if (1)
		{
			int histogram_BGR[3 * 256];
			{
				SCOPED_TIMER(histogram BGR);
				ImageAlg::histogram<3>(lenaFace.data, facePxCount, histogram_BGR);
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
			const int increment = 3; //this will skip pixels and speed up
			{
				SCOPED_TIMER(NCC Val of original and doodled);
				NCCVal = ImageAlg::NCC(lenaGray.data, lenaDGray.data, pxCount, increment);
			}
			std::cout << "NCC Val of original and doodled : " << NCCVal << std::endl;
		}

		// find face
		//  * this naive approach takes tons of time...
		//	* needs to be improved by i)parallel ii)dynamic iii)what else?
		if (0)
		{
			double* outputs = nullptr;
			const int increment = 3; //this will skip pixels and speed up
			std::pair<int, int> bestMatchPx({ -1,-1 });	// x, y = col, row
			double bestMatchValue = 0.;
			{
				SCOPED_TIMER(NCC - find face);
				ImageAlg::NCC(
					lenaGray.data, wid, hi,
					lenaFaceGray.data, faceWid, faceHi,
					outputs, increment,
					&bestMatchPx, &bestMatchValue
				);
			}
			std::cout <<
				"NCC - find face\n" <<
				"Px that best matches (Left top corner) [x,y] = [" << bestMatchPx.first << ", " << bestMatchPx.second << "]\n" <<
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
	if (1)
	{
		{
			SCOPED_TIMER(gaussian convolution);
			ImageAlg::gaussianConvolution(lenaGray.data, lena_gauss.data, wid, hi, 5, 2.0);
		}
		cv::imwrite("lena_gauss.jpg", lena_gauss);
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

	//dilation
	cv::Mat DoG_dilated(hi, wid, CV_8U);
	if (1)
	{
		const int dilation_kern_size = 5; // operate with 5x5 kernel
		{
			SCOPED_TIMER(Dilation of DoG);
			ImageAlg::dilation(DoG_bin.data, DoG_dilated.data, wid, hi, dilation_kern_size);
		}
		cv::imwrite("lena_DoG_dilation.jpg", DoG_dilated);
	}

	//synth image
	cv::Mat synth;
	lenaBGR.copyTo(synth);
	if (1)
	{
		{
			SCOPED_TIMER(Synth image);
			for (int y = 0; y < hi; ++y)
				for (int x = 0; x < wid; ++x)
					if (!DoG_dilated.at<uint8_t>(y, x))
						synth.at<cv::Vec3b>(y, x) = synth.at<cv::Vec3b>(y, x) *0.3;
		}
		cv::imwrite("lena_synthed.jpg", synth);
	}

	//image transformation
	std::shared_ptr<cv::Mat> transformed[8];
	for (int idx = 0; idx < 4; ++idx)
		transformed[idx] = std::make_shared<cv::Mat>(hi, wid, CV_8UC3);
	for (int idx = 4; idx < 8; ++idx)
		transformed[idx] = std::make_shared<cv::Mat>(wid, hi, CV_8UC3);
	if (1)
	{
#if 0
		// pass transform type as a parameter
		for (int idx = 0; idx < 8; ++idx)
		{
			{
				SCOPED_TIMER(transformation);
				ImageAlg::transformation((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[idx]->data, wid, hi, ImageAlg::ImageTransform(idx));
			}
			cv::imwrite(std::string("lena_transformed") + std::to_string(idx) + std::string(".jpg"), *transformed[idx]);
		}
#else
		// use template function for transformation
		using TX = ImageAlg::ImageTransform;
		{
			SCOPED_TIMER(transformation);
			ImageAlg::transformation<TX::None>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[0]->data, wid, hi);
		}
		{
			SCOPED_TIMER(transformation);
			ImageAlg::transformation<TX::MirrorLeftRight>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[1]->data, wid, hi);
		}
		{
			SCOPED_TIMER(transformation);
			ImageAlg::transformation<TX::MirrorUpDown>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[2]->data, wid, hi);
		}
		{
			SCOPED_TIMER(transformation);
			ImageAlg::transformation<TX::Rotate180>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[3]->data, wid, hi);
		}
		{
			SCOPED_TIMER(transformation);
			ImageAlg::transformation<TX::Rotate90CW>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[4]->data, wid, hi);
		}
		{
			SCOPED_TIMER(transformation);
			ImageAlg::transformation<TX::Rotate90CCW>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[5]->data, wid, hi);
		}
		{
			SCOPED_TIMER(transformation);
			ImageAlg::transformation<TX::Rotate90CW_MirrorUpDown>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[6]->data, wid, hi);
		}
		{
			SCOPED_TIMER(transformation);
			ImageAlg::transformation<TX::Rotate90CCW_MirrorUpDown>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[7]->data, wid, hi);
		}
		//for (int idx = 0; idx < 8; ++idx)
		//	cv::imwrite(std::string("lena_transformed") + std::to_string(idx) + std::string(".jpg"), *transformed[idx]);
#endif
	}

	std::cout << "Program finished" << std::endl;
}
