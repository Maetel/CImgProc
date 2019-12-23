//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////
#include "JamExecution.h"

#include "CImgProc.h"
#include "JamUtility.h"
#include "CLDifference.h"
#include "CLConvert2Gray.h"
#include "CLConvolution.h"
#include "CLDiffuse.h"

//std
#include <functional>

//////3rd party
//opencv
#include "opencv2/highgui.hpp"
//eigen
#include "eigen.h"
//HttpsLib
#ifdef CIMG_LINK_HTTPLIB
#include "httplib.h"
#endif
//Pico json
#ifdef CIMG_LINK_PICOJSON
#include "picojson.h"
#endif

namespace CIMGPROC
{
#define DEF_LOAD_IMAGE(_ch) \
bool Jam::_loadImage##_ch(std::string const& path, uint8_t*& data, int& wid, int& hi) \
{ \
	cv::samples::addSamplesDataSearchPath(RESOURCES_DIR); \
	auto img = cv::imread(cv::samples::findFile(path)); \
	if (img.empty()) \
		return false; \
	if (data) \
		delete data; \
	const int _wid = img.cols, _hi = img.rows; \
	data = new uint8_t[_wid * _hi * _ch]; \
	memcpy(data, img.data, _wid * _hi * _ch); \
	wid = _wid; \
	hi = _hi; \
	return true; \
}

	DEF_LOAD_IMAGE(1);
	DEF_LOAD_IMAGE(2);
	DEF_LOAD_IMAGE(3);
	DEF_LOAD_IMAGE(4);


	bool Jam::loadBGRandMakeGray(std::string const& path, uint8_t*& bgr, uint8_t*& gray, int& wid, int& hi)
	{
		if (!_loadImage3(path, bgr, wid, hi))
			return false;
		
		if (gray)
			delete gray;
		
		gray = new uint8_t[wid * hi];
		ImageAlg::convert2Gray<ImageAlg::BGR2GRAY>(bgr, gray, wid * hi);

		return true;
	}

	void Jam::loadImageTester()
	{
		const std::string path = "lena.jpg";
		uint8_t* bgr = 0, *gray = 0;
		int wid, hi;
		if (!loadBGRandMakeGray(path, bgr, gray, wid, hi))
		{
			std::cout << "No image or image path not correct" << std::endl;
			return;
		}
		
		cv::Mat bgrImg(hi, wid, CV_8UC3, bgr);
		cv::Mat grayImg(hi, wid, CV_8U, gray);

		{
			//write loaded & made image
			cv::imwrite("bgrLoaded.bmp", bgrImg);
			cv::imwrite("bgr2Gray.bmp", grayImg);
		}
	}

	void Jam::convert2Gray()
	{
		uint8_t* lena = 0;
		int wid, hi;
		if (!loadImage(std::string("lena.jpg"), lena, wid, hi))
			return;

		uint8_t* gray = new uint8_t[wid * hi];
		ImageAlg::convert2Gray<ImageAlg::BGR2GRAY>(lena, gray, wid * hi);

		{
			//show with cv::Mat
			cv::Mat grayImg(hi, wid, CV_8U, gray);
			cv::imwrite("lenaGray.bmp", grayImg);
		}
		
		delete lena;
		delete gray;
	}

	void Jam::histogram()
	{
		const std::string path = "lena.jpg";
		uint8_t* lena = 0, * gray = 0;
		int wid, hi;
		if (!loadBGRandMakeGray(path, lena, gray, wid, hi))
		{
			std::cout << "No image or image path not correct" << std::endl;
			return;
		}
		const int pxCount = wid * hi;

		auto speakHistogram = [](int const* histogram) 
		{
			for (int intensity = 0; intensity < 256; ++intensity)
				std::cout << "Intensity[" << intensity << "] value[" << histogram[intensity] << "]" << std::endl;
		};

		//gray
		int histogram_gray[256];
		{
			Util::SCOPED_TIMER(histogram gray);
			ImageAlg::histogram(gray, pxCount, histogram_gray);
		}
		std::cout << "========================================================================" << std::endl;
		std::cout << " [Histogram - gray]" << std::endl;
		speakHistogram(histogram_gray);

		//BGR
		int histogram_BGR[3 * 256];
		{
			Util::SCOPED_TIMER(histogram BGR);
			ImageAlg::histogram<3>(lena, pxCount, histogram_BGR);
		}
		int histogram_B[256];
		int histogram_G[256];
		int histogram_R[256];
		ImageAlg::extractChannel<0, 3>(histogram_BGR, histogram_B, 256);
		ImageAlg::extractChannel<1, 3>(histogram_BGR, histogram_G, 256);
		ImageAlg::extractChannel<2, 3>(histogram_BGR, histogram_R, 256);
		std::cout << "========================================================================" << std::endl;
		std::cout << " [Histogram - B]" << std::endl;
		speakHistogram(histogram_B);
		std::cout << " [Histogram - G]" << std::endl;
		speakHistogram(histogram_G);
		std::cout << " [Histogram - R]" << std::endl;
		speakHistogram(histogram_R);
		
		delete[] lena;
		delete[] gray;
	}

	void Jam::NCC()
	{
		uint8_t *lenaBGR = 0, *lenaGray = 0;
		uint8_t *lenaDdlBGR = 0, *lenaDdlGray = 0; // doodled
		uint8_t* lenaFaceBGR = 0, *lenaFaceGray = 0;
		int wid, hi, wid_ddl, hi_ddl;
		int wid_face, hi_face;
		loadBGRandMakeGray("lena.jpg", lenaBGR, lenaGray, wid, hi);
		loadBGRandMakeGray("lena_doodled.jpg", lenaDdlBGR, lenaDdlGray, wid_ddl, hi_ddl);
		loadBGRandMakeGray("lena_face.png", lenaFaceBGR, lenaFaceGray, wid_face, hi_face);
		if (wid != wid_ddl || hi != hi_ddl)
			return;
		const int pxCount = wid * hi;


		double NCCVal;
		//of the same size
		{
			const int increment = 3; //this will skip pixels and speed up
			Util::SCOPED_TIMER(NCC Val of original and doodled);
			NCCVal = ImageAlg::NCC(lenaGray, lenaDdlGray, pxCount, increment);
		}
		std::cout << "NCC Val of original and doodled : " << NCCVal << std::endl;

		// find face
		//  * this naive approach takes tons of time...
		//	* needs to be improved by i)parallel ii)dynamic iii)what else?
		double* outputs = nullptr;
		std::pair<int, int> bestMatchPx({ -1,-1 });	// x, y = col, row
		double bestMatchValue = 0.;
		{
			const int increment = 3; //this will skip pixels and speed up
			Util::SCOPED_TIMER(NCC - find face);
			ImageAlg::NCC(
				lenaGray, wid, hi,
				lenaFaceGray, wid_face, hi_face,
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

		delete[] lenaBGR;
		delete[] lenaGray;
		delete[] lenaDdlBGR;
		delete[] lenaDdlGray;
	}

	void Jam::ZNCC()
	{
		uint8_t* lenaBGR = 0, * lenaGray = 0;
		uint8_t* lenaDdlBGR = 0, * lenaDdlGray = 0; // doodled
		int wid, hi, wid_ddl, hi_ddl;
		int wid_face, hi_face;
		loadBGRandMakeGray("lena.jpg", lenaBGR, lenaGray, wid, hi);
		loadBGRandMakeGray("lena_doodled.jpg", lenaDdlBGR, lenaDdlGray, wid_ddl, hi_ddl);
		if (wid != wid_ddl || hi != hi_ddl)
			return;
		const int pxCount = wid * hi;

		double ZNCCVal = HUGE_VAL;
		{
			Util::SCOPED_TIMER(ZNCC);
			ZNCCVal = ImageAlg::ZNCC(lenaGray, lenaDdlGray, pxCount);
		}
		std::cout << "ZNCC value : " << ZNCCVal << std::endl;

		delete[] lenaBGR;
		delete[] lenaGray;
		delete[] lenaDdlBGR;
		delete[] lenaDdlGray;
	}

	void Jam::binarization()
	{
		uint8_t* lenaBGR = 0, * lenaGray = 0;
		int wid, hi;
		if (!loadBGRandMakeGray("lena.jpg", lenaBGR, lenaGray, wid, hi))
			return;
		const int pxCount = wid * hi;

		uint8_t* lena_binarized = new uint8_t[pxCount];
		double otsu_thres;
		{
			Util::SCOPED_TIMER(Otsu binarization);
			ImageAlg::binarize(lenaGray, lena_binarized, pxCount, &otsu_thres);
		}
		cv::Mat binarizedImg(hi, wid, CV_8U, lena_binarized);
		cv::imwrite("lena_otsu.bmp", binarizedImg);
		std::cout << "Otsu binarization threshold : " << otsu_thres << std::endl;

		{
			Util::SCOPED_TIMER(Threshold binarization);
			int binThres = 128;
			ImageAlg::binarize<ImageAlg::THRESHOLD>(lenaGray, lena_binarized, pxCount, &binThres);
		}
		cv::imwrite("lena_binarized.jpg", binarizedImg);

		delete[] lenaBGR;
		delete[] lenaGray;
		delete[] lena_binarized;
	}

	void Jam::gaussian()
	{
		uint8_t* lenaBGR = 0, * lenaGray = 0;
		int wid, hi;
		if (!loadBGRandMakeGray("lena.jpg", lenaBGR, lenaGray, wid, hi))
			return;
		const int pxCount = wid * hi;

		uint8_t* lena_gaussed = new uint8_t[pxCount];
		const int kern_size = 7;
		const double gaussian_sigma = 2.0;
		{
			Util::SCOPED_TIMER(gaussian convolution);
			ImageAlg::gaussianConvolution(lenaGray, lena_gaussed, wid, hi, kern_size, gaussian_sigma);
		}
		cv::Mat lena_gauss(hi, wid, CV_8U, lena_gaussed);
		cv::imwrite("lena_gauss.jpg", lena_gauss);
		
		delete[] lenaBGR;
		delete[] lenaGray;
		delete[] lena_gaussed;
	}

	void Jam::derivation()
	{
		uint8_t* lenaBGR = 0, * lenaGray = 0;
		int wid, hi;
		if (!loadBGRandMakeGray("lena.jpg", lenaBGR, lenaGray, wid, hi))
			return;
		const int pxCount = wid * hi;

		cv::Mat derivedImg;
		float* lena_derived = new float[pxCount];
		
		//sobel dx
		{
			Util::SCOPED_TIMER(sobel dx);
			ImageAlg::sobelDx(lenaGray, lena_derived, wid, hi);
		}
		derivedImg = cv::Mat(hi, wid, CV_32F, lena_derived);
		cv::imwrite("lena_sobelDx.jpg", derivedImg);

		//sobel dy
		{
			Util::SCOPED_TIMER(sobel dy);
			ImageAlg::sobelDy(lenaGray, lena_derived, wid, hi);
		}
		derivedImg = cv::Mat(hi, wid, CV_32F, lena_derived);
		cv::imwrite("lena_sobelDy.jpg", derivedImg);

		//scharr dx
		{
			Util::SCOPED_TIMER(scharr dx);
			ImageAlg::scharrDx(lenaGray, lena_derived, wid, hi);
		}
		derivedImg = cv::Mat(hi, wid, CV_32F, lena_derived);
		cv::imwrite("lena_scharrDx.jpg", derivedImg);

		//scharr dy
		{
			Util::SCOPED_TIMER(scharr dy);
			ImageAlg::scharrDy(lenaGray, lena_derived, wid, hi);
		}
		derivedImg = cv::Mat(hi, wid, CV_32F, lena_derived);
		cv::imwrite("lena_scharrDy.jpg", derivedImg);

		delete[] lenaBGR;
		delete[] lenaGray;
		delete[] lena_derived;
	}

	void Jam::convolution()
	{
		uint8_t* lenaBGR = 0, * lenaGray = 0;
		int wid, hi;
		if (!loadBGRandMakeGray("lena.jpg", lenaBGR, lenaGray, wid, hi))
			return;
		const int pxCount = wid * hi;

		uint8_t* lena_convo= new uint8_t[pxCount];
		constexpr int convoKernel[] =
		{
			1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 0, 0, 0, 0, 0, 0, 0, 1,
			1, 0, 0, 0, 0, 0, 0, 0, 1,
			1, 0, 0, 0, -3, 0, 0, 0, 1,
			1, 0, 0, -3, 0, -3, 0, 0, 1,
			1, 0, 0, 0, -3, 0, 0, 0, 1,
			1, 0, 0, 0, 0, 0, 0, 0, 1,
			1, 0, 0, 0, 0, 0, 0, 0, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1
		};

		{
			Util::SCOPED_TIMER(convo);
			ImageAlg::convolution(lenaGray, lena_convo, wid, hi, convoKernel, 9, 9);
		}
		cv::Mat convoImg(hi, wid, CV_8U, lena_convo);
		cv::imwrite("lena_convo.bmp", convoImg);

		delete[] lenaBGR;
		delete[] lenaGray;
		delete[] lena_convo;
	}

	void Jam::colorMagnet()
	{
		uint8_t* lenaBGR = 0, *lenaGray = 0;
		int wid, hi;
		if (!loadBGRandMakeGray("lena.jpg", lenaBGR, lenaGray, wid, hi))
			return;
		const int pxCount = wid * hi;

		constexpr auto colorMagnetChannel = 3;
		uint8_t* lena_magnet = new uint8_t[pxCount * colorMagnetChannel];

		// color combo from https://www.designwizard.com/blog/design-trends/colour-combination
		// color order in BGR
		uint8_t dstColors[] =
		{
#if 1
			0xbd, 0x3c, 0x96,
			0x61, 0x6f, 0xff,
			0x9b, 0x29, 0xc5,
			0x51, 0xae, 0xfe,
#else
			0xe2, 0xdd, 0xff,
			0x94, 0xa0, 0xfa,
			0xcc, 0xd9, 0x9e,
			0x76, 0x8c, 0x00
#endif
		};

		// 'dstColorSize' will be divided by 'Channel' inside the function for computation
		const auto dstColorsSize = sizeof(dstColors) / sizeof(uint8_t); 
		constexpr int Channel = 3;
		{
			//8-9ms in release
			Util::SCOPED_TIMER(Color magnet);
			CIMGPROC::ImageAlg::colorMagnet<Channel>(lenaBGR, lena_magnet, wid * hi, dstColors, dstColorsSize);
		}

		cv::Mat magnetImg(hi, wid, CV_8UC3, lena_magnet);
		cv::imwrite("lena_magnet.jpg", magnetImg);

		delete[] lenaBGR;
		delete[] lenaGray;
		delete[] lena_magnet;
	}

	void Jam::execute()
	{
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
			Util::SCOPED_TIMER(Lena BGR2Gray);
			ImageAlg::convert2Gray<ImageAlg::RGB2GRAY>(lenaBGR.data, lenaGray.data, pxCount);
		}
		cv::imwrite("lena_gray.jpg", lenaGray);

		cv::Mat lenaDGray(hi, wid, CV_8U);
		{
			Util::SCOPED_TIMER(Lena Doodled BGR2Gray);
			ImageAlg::convert2Gray<ImageAlg::BGR2GRAY>(lenaDoodled.data, lenaDGray.data, pxCount);
		}
		cv::imwrite("lena_doodled_gray.jpg", lenaDGray);

		cv::Mat lenaFaceGray(faceHi, faceWid, CV_8U);
		{
			Util::SCOPED_TIMER(Lena Doodled BGR2Gray);
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
					Util::SCOPED_TIMER(histogram gray);
					ImageAlg::histogram(lenaFaceGray.data, facePxCount, histogram_gray);
				}
				//std::cout << histogram_gray << std::endl;
			}
			//BGR
			if (1)
			{
				int histogram_BGR[3 * 256];
				{
					Util::SCOPED_TIMER(histogram BGR);
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
					Util::SCOPED_TIMER(NCC Val of original and doodled);
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
					Util::SCOPED_TIMER(NCC - find face);
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
				Util::SCOPED_TIMER(ZNCC);
				ZNCCVal = ImageAlg::ZNCC(lenaGray.data, lenaDGray.data, pxCount);
			}
			std::cout << "ZNCC value : " << ZNCCVal << std::endl;
		}

		//binarization
		cv::Mat lena_otsu(hi, wid, CV_8U);
		if (1)
		{
			{
				Util::SCOPED_TIMER(Otsu binarization);
				ImageAlg::binarize(lenaGray.data, lena_otsu.data, pxCount);
				//int binThres = 50;
				//ImageAlg::binarize<ImageAlg::THRESHOLD>(lenaGray.data, lena_binarized.data, pxCount, &binThres);

			}
			cv::imwrite("lena_Otsu.jpg", lena_otsu);
		}

		//gaussian convolution
		cv::Mat lena_gauss(hi, wid, CV_8U);
		if (1)
		{
			{
				Util::SCOPED_TIMER(gaussian convolution);
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
				Util::SCOPED_TIMER(convolution(sobel));
				ImageAlg::convolution(lena_gauss.data, (short*)lena_sobel.data, wid, hi, sobel.data(), 3, 3);
			}
			cv::imwrite("lena_sobel.jpg", lena_sobel);
		}

		//difference of gaussian
		cv::Mat DoG_bin(hi, wid, CV_8U);
		if (1)
		{
			cv::Mat DoG(hi, wid, CV_8U);
			{
				Util::SCOPED_TIMER(difference of gaussian);
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
				Util::SCOPED_TIMER(Dilation of DoG);
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
				Util::SCOPED_TIMER(Synth image);
				for (int y = 0; y < hi; ++y)
					for (int x = 0; x < wid; ++x)
						if (!DoG_dilated.at<uint8_t>(y, x))
							synth.at<cv::Vec3b>(y, x) = synth.at<cv::Vec3b>(y, x) * 0.3;
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
					Util::SCOPED_TIMER(transformation);
					ImageAlg::transformation((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[idx]->data, wid, hi, ImageAlg::ImageTransform(idx));
				}
				cv::imwrite(std::string("lena_transformed") + std::to_string(idx) + std::string(".jpg"), *transformed[idx]);
			}
#else
			// use template function for transformation
			using TX = ImageAlg::ImageTransform;
			{
				Util::SCOPED_TIMER(transformation);
				ImageAlg::transformation<TX::None>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[0]->data, wid, hi);
			}
			{
				Util::SCOPED_TIMER(transformation);
				ImageAlg::transformation<TX::MirrorLeftRight>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[1]->data, wid, hi);
			}
			{
				Util::SCOPED_TIMER(transformation);
				ImageAlg::transformation<TX::MirrorUpDown>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[2]->data, wid, hi);
			}
			{
				Util::SCOPED_TIMER(transformation);
				ImageAlg::transformation<TX::Rotate180>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[3]->data, wid, hi);
			}
			{
				Util::SCOPED_TIMER(transformation);
				ImageAlg::transformation<TX::Rotate90CW>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[4]->data, wid, hi);
			}
			{
				Util::SCOPED_TIMER(transformation);
				ImageAlg::transformation<TX::Rotate90CCW>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[5]->data, wid, hi);
			}
			{
				Util::SCOPED_TIMER(transformation);
				ImageAlg::transformation<TX::Rotate90CW_MirrorUpDown>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[6]->data, wid, hi);
			}
			{
				Util::SCOPED_TIMER(transformation);
				ImageAlg::transformation<TX::Rotate90CCW_MirrorUpDown>((cv::Vec3b*)lenaBGR.data, (cv::Vec3b*)transformed[7]->data, wid, hi);
			}
			//for (int idx = 0; idx < 8; ++idx)
			//	cv::imwrite(std::string("lena_transformed") + std::to_string(idx) + std::string(".jpg"), *transformed[idx]);
#endif
		}

		//median filter
		cv::Mat lena_median(hi, wid, CV_8U);
		if (1)
		{
			const int kernel_size = 3;
			{
				Util::SCOPED_TIMER(median filter);
				ImageAlg::medianFilter(lenaGray.data, lena_median.data, wid, hi, kernel_size);
			}
			cv::imwrite("lena_median.jpg", lena_median);
		}

		std::cout << "Program finished" << std::endl;
	}

	void Jam::runCL()
	{
		cv::samples::addSamplesDataSearchPath(RESOURCES_DIR);
		std::string lenaPath("lena.jpg");					//RGB image
		auto lenaBGR = cv::imread(cv::samples::findFile(lenaPath));
		const int wid = lenaBGR.cols, hi = lenaBGR.rows, pxCount = wid * hi;

		cv::Mat lenaGray(hi, wid, CV_8U);
		//to gray
		ImageAlg::convert2Gray<ImageAlg::BGR2GRAY>(lenaBGR.data, lenaGray.data, pxCount);

		//prepare cl buffers
		auto context = CLInstance.context();
		auto queue = cl::CommandQueue(context);
		{
			//show info
			std::cout <<
				"OpenCL Info : \n" <<
				"Platform info : " << CLInstance.platformInfo() << "\n" <<
				"Device info : " << CLInstance.deviceInfo() <<
				std::endl;
		}
		
		
		cl::Buffer input_BGR(context, CL_MEM_COPY_HOST_PTR, pxCount * sizeof(uint8_t) * 3, lenaBGR.data);
		cl::Buffer input_gray(context, CL_MEM_COPY_HOST_PTR, pxCount * sizeof(uint8_t), lenaGray.data);
		cl::Buffer input_l(context, CL_MEM_COPY_HOST_PTR, pxCount * sizeof(uint8_t), lenaGray.data);
		cl::Buffer input_r(context, CL_MEM_COPY_HOST_PTR, pxCount * sizeof(uint8_t), lenaGray.data);
		cl::Buffer output(context, CL_MEM_WRITE_ONLY, pxCount * sizeof(uint8_t));

		//cl difference
		{
			CL::CLDifference diff;
			diff.build();
			{
				Util::SCOPED_TIMER(opencl difference);
				diff.difference(queue, input_l, input_r, output, pxCount);
			}
		}
		
		//cl rgb2gray
		{
			CL::CLConvert2Gray clcvt2gray;
			clcvt2gray.build();
			{
				Util::SCOPED_TIMER(opencl cvt2gray);
				clcvt2gray.convert2gray(queue, input_BGR, output, pxCount, CIMGPROC::CL::CLConvert2Gray::BGR2GRAY);
			}
		}
		
		//cl convolution
		{
			cl::Buffer convoOutput(context, CL_MEM_WRITE_ONLY, pxCount * sizeof(uint8_t));
			cv::Mat convoImg(hi, wid, CV_8U);

			CL::CLConvolution convolution;
			{
				Util::SCOPED_TIMER(cl convolution build);
				convolution.build();
			}

			//gaussian
			{
				{
					Util::SCOPED_TIMER(cl gaussian);
					convolution.gaussianConvolution(queue, input_gray, convoOutput, wid, hi, 5, 1.0);
				}
				CL::download(queue, convoOutput, convoImg.data, pxCount);
				cv::imwrite("lena_cl_gauss.jpg", convoImg);
			}

			//derivative
			{
				{
					Util::SCOPED_TIMER(cl sobel dx);
					convolution.sobel_dx(queue, input_gray, convoOutput, wid, hi);
				}
				CL::download(queue, convoOutput, convoImg.data, pxCount);
				cv::imwrite("lena_cl_sobel.jpg", convoImg);
			}

			//custom kernel
			{
				std::vector<float> glassFrame =
				{
					1, 1, 1, 1, 1, 1, 1,
					1, 0, 0, 0, 0, 0, 1,
					1, 0, -3, -7, -3, 0, 1,
					1, 0, -7, 30, -7, 0, 1,
					1, 0, -3, -7, -3, 0, 1,
					1, 0, 0, 0, 0, 0, 1,
					1, 1, 1, 1, 1, 1, 1
				};

				CL::KernelBuffer glassBuffer(context, glassFrame.data(), 7, 7, true);
				{
					Util::SCOPED_TIMER(cl glass effect);
					convolution.convolution(queue, input_gray, convoOutput, wid, hi, glassBuffer);
				}

				CL::download(queue, convoOutput, convoImg.data, pxCount);
				cv::imwrite("lena_cl_glass.jpg", convoImg);
			}
		}

		cv::Mat runByCL(hi, wid, CV_8U);
		CL::download(queue, output, runByCL.data, pxCount);
		cv::imwrite("lena_brightend_cl.jpg", runByCL);


		std::cout << "Program finished" << std::endl;
	}

	//this function's not ready for exception handling
	std::tuple<int, int, int, int> parseAzureFaceJson (std::string const& input)
	{
		const auto faceRectCategory = input.find(std::string(R"("faceRectangle")"));
		const auto faceRectStart = input.find("{", faceRectCategory);
		const auto faceRectEnd = input.find("}", faceRectCategory);
		const auto faceRect = input.substr(faceRectStart, faceRectEnd + 1 - faceRectStart);
		const std::string leftStr(R"("left":)");
		const std::string topStr(R"("top":)");
		const std::string widthStr(R"("width":)");
		const std::string heightStr(R"("height":)");

		// TODO : replace with regex
		//	std::regex end("^(,|})$");
		std::string end(",");
		const auto leftPos = faceRect.find(leftStr) + leftStr.size();
		const auto topPos = faceRect.find(topStr) + topStr.size();
		const auto widPos = faceRect.find(widthStr) + widthStr.size();
		const auto hiPos = faceRect.find(heightStr) + heightStr.size();
		const auto leftEnd = faceRect.find(end, leftPos);
		const auto topEnd = faceRect.find(end, topPos);
		const auto widEnd = faceRect.find(end, widPos);
		const auto hiEnd = faceRect.find("}", hiPos);

		const auto leftS = faceRect.substr(leftPos, leftEnd - leftPos);
		const auto topS = faceRect.substr(topPos, topEnd - topPos);
		const auto widS = faceRect.substr(widPos, widEnd - widPos);
		const auto hiS = faceRect.substr(hiPos, hiEnd - hiPos);

		return { std::stoi(leftS), std::stoi(topS), std::stoi(widS), std::stoi(hiS) };
	}

	//test MS Azure Face API
	void Jam::runHttpsClient()
	{
#if defined(CIMG_LINK_HTTPLIB) // && (CIMG_LINK_PICOJSON)
#ifndef CIMG_FACE_API_KEY
#error(define your own Face API key path)
#endif

		//test face api
		httplib::Client cli("cimgproc-test1.cognitiveservices.azure.com");
		std::shared_ptr<httplib::Response> response = 0;

		int try_count = 5;
		while (try_count-- > 0)
		{
			std::string jam_face_api_key = Util::fileToStr(CIMG_FACE_API_KEY);
			httplib::Headers headers({{"Ocp-Apim-Subscription-Key" , jam_face_api_key }});
			std::string body(R"({"url" : "https://github.com/Maetel/CImgProc/blob/master/resources/lena_doodled.jpg?raw=true"})");
				
			response = cli.Post(
				"/vision/v2.0/analyze?visualFeatures=Faces&details=Landmarks&language=en",
				headers,
				body,
				"application/json"
			);

			if (response)
			{
				std::cout << "Response found" << std::endl;
				break;
			}
			else
			{
				std::cout << "Trying to reconnect : " << try_count << std::endl;;
				Sleep(1000);
				continue;
			}
		}

		if (!response)
		{
			std::cerr << "No response. Returning..." << std::endl;
			return;
		}
		else if (response->status != 200)
		{
			std::cerr << "Response status wrong. Returning..." << std::endl;
			return;
		}

		//copy for manipulation and free response instance
		const auto res_body = response->body;
		response = nullptr;

		struct Point
		{
			Point() = default;
			Point(int _x, int _y) : x(_x), y(_y) {}
			~Point() {}
			int x, y;

			std::string toString() const
			{
				std::ostringstream stream;
				stream << "[x,y] = [" << x << "," << y << "]";
				return stream.str();
			}

			double distanceFrom(Point const& other) const
			{
				return std::sqrt((other.x - x) * (other.x - x) + (other.y - y) * (other.y - y));
			}
			double distanceFrom(int x, int y) const
			{
				return std::sqrt((this->x - x) * (this->x - x) + (this->y - y) * (this->y - y));
			}

			static double distanceFrom(Point const& left, Point const& right)
			{
				return std::sqrt((left.x - right.x) * (left.x - right.x) + (left.y - right.y) * (left.y - right.y));
			}
		};

		

		struct FaceRect
		{
		public:
			FaceRect() = default;
			FaceRect(std::string const& input)
			{
				auto result = parseAzureFaceJson(input);
				left = std::get<0>(result);
				top  = std::get<1>(result);
				wid  = std::get<2>(result);
				hi   = std::get<3>(result);
			}
			~FaceRect() {}

		public:
			int left = -1;
			int top = -1;
			int wid = -1;
			int hi = -1;

		public:
			
			//returns {x,y}
			Point leftTop() const { return Point(left, top); }
			Point rightTop() const { return Point(left + wid, top); }
			Point leftBottom() const { return Point(left, top + hi); }
			Point rightBottom() const { return Point(left + wid, top + hi); }
			Point centerPoint() const { 
				return Point(left + (wid / 2), top + (hi / 2));
			}
			double farthest() const
			{
				return Point::distanceFrom(leftTop(), centerPoint());
			}

			bool containsPoint(Point const& point)
			{
				return 
					(point.x >= leftTop().x) && (point.x < rightBottom().x) &&
					(point.y >= leftTop().y) && (point.y < rightBottom().y);
			}

			std::string toString() const
			{
				std::ostringstream stream;
				stream << "[Left, Top, Width, Height] = [" << left << ", " << top << ", " << wid << "," << hi << "]";
				return stream.str();
			}
		};

		FaceRect face(res_body);
		std::cout << face.toString() << std::endl;
		
		cv::samples::addSamplesDataSearchPath(RESOURCES_DIR);
		std::string lenaPath("lena.jpg");					//RGB image
		std::string lenaPath_d("lena_doodled.jpg");					//RGB image
		auto lenaBGR = cv::imread(cv::samples::findFile(lenaPath));
		auto lenaBGR_d = cv::imread(cv::samples::findFile(lenaPath_d));
		const int wid = lenaBGR.cols, hi = lenaBGR.rows;
		cv::Mat lenaGray(wid, hi, CV_8U);
		cv::Mat lenaGray_d(wid, hi, CV_8U);
		CIMGPROC::ImageAlg::convert2Gray<CIMGPROC::ImageAlg::BGR2GRAY>(lenaBGR.data, lenaGray.data, wid* hi);
		CIMGPROC::ImageAlg::convert2Gray<CIMGPROC::ImageAlg::BGR2GRAY>(lenaBGR_d.data, lenaGray_d.data, wid* hi);
		
		cv::Mat lenaGrayFaceDetected(wid, hi, CV_8U);
		cv::Mat lena_mask(wid, hi, CV_32F);
		lena_mask.setTo(0);

		const auto faceCenterPoint = face.centerPoint();
		const auto faceRectDistance = face.farthest();
		for (int y = 0; y < hi; ++y)
		{
			for (int x = 0; x < wid; ++x)
			{
				const Point curPoint(x, y);
#if 1
				const auto dist = curPoint.distanceFrom(faceCenterPoint);
				if (dist > faceRectDistance)
					continue;

				lena_mask.at<float>(y, x) = float(1 - dist / faceRectDistance);
#else
				if (face.containsPoint(curPoint))
					lena_mask.at<float>(y, x) = 1;
#endif
			}
		}

		for (int idx = 0; idx < wid * hi; ++idx)
		{
			lenaGrayFaceDetected.data[idx] = ((float const*)lena_mask.data)[idx] * (lenaGray.data[idx]);
		}

		cv::imwrite("lenaGrayFaceDetected.jpg", lenaGrayFaceDetected);

		//median for diffuse
		cv::Mat medianed(wid, hi, CV_8U);
		if(0)
		{
			{
				//median for diffuse
				Util::SCOPED_TIMER(Median for diffuse);
				CIMGPROC::ImageAlg::medianFilter(lenaGray.data, medianed.data, wid, hi, 15);
			}
			cv::imwrite("lena_medianed.jpg", medianed);
		}
		else
		{
			//load premaid
			medianed = cv::imread("lena_medianed.jpg", cv::IMREAD_GRAYSCALE);
		}

		//diffuse
		{
			cv::Mat diffused(wid, hi, CV_8U);
			{
				//diffuse with ratio
				Util::SCOPED_TIMER(diffuse with ratio);
				CIMGPROC::ImageAlg::diffuse(lenaGray.data, lenaGrayFaceDetected.data, 0.5, diffused.data, wid* hi);
			}
			cv::imwrite("lenaDiffused_ratio.jpg", diffused);

			{
				//diffuse with mask
				Util::SCOPED_TIMER(diffuse with mask);
				CIMGPROC::ImageAlg::diffuse(medianed.data, lenaGray.data, (float const*)lena_mask.data, diffused.data, wid * hi);
			}
			cv::imwrite("lenaDiffused_mask.jpg", diffused);
		}

		//prepare cl buffers
		auto context = CLInstance.context();
		auto queue = cl::CommandQueue(context);

		//diffuse with CL
		{
			cl_int err;
			cl::Buffer grayBuf(context, CL_MEM_READ_ONLY, wid* hi);
			cl::Buffer medianBuf(context, CL_MEM_READ_ONLY, wid* hi);
			cl::Buffer maskBuf(context, CL_MEM_READ_ONLY, wid* hi * sizeof(float));
			cl::Buffer diffuseBuf(context, CL_MEM_WRITE_ONLY, wid* hi);

			queue.enqueueWriteBuffer(grayBuf, false, 0, wid* hi, lenaGray.data);
			queue.enqueueWriteBuffer(medianBuf, false, 0, wid* hi, medianed.data);
			queue.enqueueWriteBuffer(maskBuf, true, 0, wid* hi * sizeof(float), lena_mask.data);
			queue.flush();
			queue.finish();

			CIMGPROC::CL::CLDiffuse diffuser;
			diffuser.build();
			cv::Mat diffused(wid, hi, CV_8U);
			for(int idx = 0; idx < 10; ++idx)
			{
				//diffuse with mask cl
				Util::SCOPED_TIMER(diffuse with mask cl);
				diffuser.diffuse(queue, grayBuf, medianBuf, diffuseBuf, maskBuf, wid* hi);
			}
			CIMGPROC::CL::download(queue, diffuseBuf, diffused.data, wid* hi);
			cv::imwrite("lenaDiffused_mask_cl.jpg", diffused);

			for (int idx = 0; idx < 10; ++idx)
			{
				//diffuse with mask cl
				Util::SCOPED_TIMER(diffuse with ratio cl);
				diffuser.diffuse(queue, grayBuf, medianBuf, diffuseBuf, 0.1f, wid * hi);
			}
			CIMGPROC::CL::download(queue, diffuseBuf, diffused.data, wid* hi);
			cv::imwrite("lenaDiffused_ratio_cl.jpg", diffused);
		}

#else
		std::cerr << "Library not linked. Returning..." << std::endl;
		return;
#endif
	} // ! runHttpsClient
	
}




