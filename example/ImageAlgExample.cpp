//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////
#include "ImageAlgExample.h"

#include "CImgProc.h"
#include "JamUtility.h"

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
bool ImageAlgExample::_loadImage##_ch(std::string const& path, uint8_t*& data, int& wid, int& hi) \
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


	bool ImageAlgExample::loadBGRandMakeGray(std::string const& path, uint8_t*& bgr, uint8_t*& gray, int& wid, int& hi)
	{
		constexpr int Channel = 3;
		if (!loadImage<Channel>(path, bgr, wid, hi))
			return false;
		
		if (gray)
			delete gray;
		
		gray = new uint8_t[wid * hi];
		ImageAlg::convert2Gray<ImageAlg::BGR2GRAY>(bgr, gray, wid * hi);

		return true;
	}

	void ImageAlgExample::loadImageTester()
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

	void ImageAlgExample::convert2Gray()
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

	void ImageAlgExample::histogram()
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

		auto speakHistogram = [](int const* histogram, bool skipZero = true) 
		{
			for (int intensity = 0; intensity < 256; ++intensity)
				if(const auto val = histogram[intensity]; val)
					std::cout << "Intensity[" << intensity << "] value[" << val << "]" << std::endl;
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

		constexpr int Channel = 3;
		//BGR
		int histogram_BGR[3 * 256];
		int* histogram_BGR_extracted[Channel];
		for (int ch; ch < Channel; ++ch)
			histogram_BGR_extracted[ch] = new int[256];
		{
			Util::SCOPED_TIMER(histogram BGR);
			ImageAlg::histogram<3>(lena, pxCount, histogram_BGR);
		}
		ImageAlg::extractChannelAll<3>(histogram_BGR, histogram_BGR_extracted, 256);
		std::cout << "========================================================================" << std::endl;
		std::cout << " [Histogram - B]" << std::endl;
		speakHistogram(histogram_BGR_extracted[0]);
		std::cout << " [Histogram - G]" << std::endl;
		speakHistogram(histogram_BGR_extracted[1]);
		std::cout << " [Histogram - R]" << std::endl;
		speakHistogram(histogram_BGR_extracted[2]);
		
		delete[] lena;
		delete[] gray;
		for (int ch; ch < Channel; ++ch)
			delete[] histogram_BGR_extracted[ch];
	}

	void ImageAlgExample::NCC()
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

	void ImageAlgExample::ZNCC()
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

	void ImageAlgExample::binarization()
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

	void ImageAlgExample::gaussian()
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

	void ImageAlgExample::derivation()
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

	void ImageAlgExample::convolution()
	{
		uint8_t* lenaBGR = 0, * lenaGray = 0;
		int wid, hi;
		if (!loadBGRandMakeGray("lena.jpg", lenaBGR, lenaGray, wid, hi))
			return;
		const int pxCount = wid * hi;

		uint8_t* lena_convo= new uint8_t[pxCount];
		constexpr int kern_wid = 7, kern_hi = 7;
		constexpr int convoKernel[kern_wid * kern_hi] =
		{
			1, 1, 1, 1, 1, 1, 1,
			1, 0, 0, 0, 0, 0, 1,
			1, 0, -3, -7, -3, 0, 1,
			1, 0, -7, 30, -7, 0, 1,
			1, 0, -3, -7, -3, 0, 1,
			1, 0, 0, 0, 0, 0, 1,
			1, 1, 1, 1, 1, 1, 1
		};

		{
			Util::SCOPED_TIMER(convo);
			ImageAlg::convolution(lenaGray, lena_convo, wid, hi, convoKernel, kern_wid, kern_hi);
		}
		cv::Mat convoImg(hi, wid, CV_8U, lena_convo);
		cv::imwrite("lena_convo.bmp", convoImg);

		delete[] lenaBGR;
		delete[] lenaGray;
		delete[] lena_convo;
	}

	void ImageAlgExample::colorMagnet()
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

	void ImageAlgExample::differenceOfGaussian()
	{
		uint8_t* lenaBGR = 0, *lenaGray = 0;
		int wid, hi;
		if (!loadBGRandMakeGray("lena.jpg", lenaBGR, lenaGray, wid, hi))
			return;
		const int pxCount = wid * hi;

		//difference of gaussian
		uint8_t* DoG = new uint8_t[wid * hi];
		uint8_t* DoG_bin = new uint8_t[wid * hi];
		{
			Util::SCOPED_TIMER(difference of gaussian);
			ImageAlg::differenceOfGaussian(lenaGray, DoG, wid, hi);
		}

		{
			//binarize and check result
			int binThres;
			ImageAlg::binarize(DoG, DoG_bin, pxCount, &binThres);
			cv::Mat DoGBinImg(hi, wid, CV_8U, DoG_bin);
			cv::imwrite("lena_DoG_binarized.jpg", DoGBinImg);
			std::cout << "Otsu binarization threshold : " << binThres << std::endl;
		}
		auto* toBeSynthed = DoG_bin;

		uint8_t* DoG_dilated = 0;
		if (0)
		{
			//dilation
			DoG_dilated = new uint8_t[wid * hi];
			const int dilation_kern_size = 5; // operate with 5x5 kernel
			{
				Util::SCOPED_TIMER(Dilation of DoG);
				ImageAlg::dilation(DoG_bin, DoG_dilated, wid, hi, dilation_kern_size);
			}
			toBeSynthed = DoG_dilated;

			{
				cv::Mat DoG_dilated_img(hi, wid, CV_8U, DoG_dilated);
				cv::imwrite("lena_DoG_dilation.jpg", DoG_dilated_img);
			}
		}
		
		//synth BGR image
		uint8_t* synth = new uint8_t[wid * hi * 3];
		memcpy(synth, lenaBGR, pxCount * 3 *sizeof(uint8_t));
		{
			constexpr double alpha = 0.3;
			Util::SCOPED_TIMER(Synth image);
			for(int idx = 0; idx < pxCount; ++idx)
				if (!toBeSynthed[idx])
					for(int ch = 0; ch < 3; ++ch)
						synth[idx*3 + ch] = uint8_t(synth[idx*3 + ch] * 0.3);
		}
		cv::Mat synthImg(hi, wid, CV_8UC3, synth);
		cv::imwrite("lena_synthed.jpg", synthImg);

		delete[] lenaBGR;
		delete[] lenaGray;
		delete[] DoG;
		delete[] DoG_bin;
		if(nullptr != DoG_dilated) delete[] DoG_dilated;
		delete[] synth;
	}

	void ImageAlgExample::median()
	{
		uint8_t* lenaBGR = 0, *lenaGray = 0;
		int wid, hi;
		if (!loadBGRandMakeGray("jam.jpg", lenaBGR, lenaGray, wid, hi))
			return;
		const int pxCount = wid * hi;

		constexpr int kernelSize = 11;
#if 0
		uint8_t* grayFiltered = new uint8_t[pxCount];
		{
			Util::SCOPED_TIMER(median single channel);
			ImageAlg::medianFilter(lenaGray, grayFiltered, wid, hi, kernelSize);
		}
		{
			cv::Mat grayFilteredImg(hi, wid, CV_8U, grayFiltered);
			cv::imwrite("medianGray.bmp", grayFilteredImg);
		}
		delete[] grayFiltered;
#endif

		uint8_t* bgrFiltered = new uint8_t[pxCount * 3];
		{
			Util::SCOPED_TIMER(median multi channel);
			ImageAlg::medianFilter_t<3>(lenaBGR, bgrFiltered, wid, hi, kernelSize);
		}
		{
			cv::Mat bgrFilteredImg(hi, wid, CV_8UC3, bgrFiltered);
			cv::imwrite("medianBGR.bmp", bgrFilteredImg);
		}
		delete[] bgrFiltered;


		delete[] lenaBGR;
		delete[] lenaGray;
	}

	void ImageAlgExample::extractChannel()
	{
		uint8_t* lenaBGR = 0, *lenaGray = 0;
		int wid, hi;
		if (!loadBGRandMakeGray("lena.jpg", lenaBGR, lenaGray, wid, hi))
			return;
		const int pxCount = wid * hi;

		constexpr int MaxChannel = 3;
		uint8_t* eachChannel[MaxChannel];
		for (int ch = 0; ch < MaxChannel; ++ch)
			eachChannel[ch] = new uint8_t[wid * hi];

		{
			Util::SCOPED_TIMER(extract all channel);
			ImageAlg::extractChannelAll<MaxChannel>(lenaBGR, eachChannel, pxCount);
		}

		for (int ch = 0; ch < MaxChannel; ++ch)
		{
			cv::Mat img(hi, wid, CV_8U, eachChannel[ch]);
			const std::string 
				imgName = "channelImage",
				curCh = std::to_string(ch),
				suffix = ".bmp";
			cv::imwrite(imgName + curCh + suffix, img); // "channelImage0.bmp"
		}
		for (int ch = 0; ch < MaxChannel; ++ch)
			delete eachChannel[ch];

		delete[] lenaBGR;
		delete[] lenaGray;
	}

	//detect face -> apply median filter for non-face region -> fade out from face center
	void ImageAlgExample::faceDetectionAndManipulation()
	{
#if defined(CIMG_LINK_HTTPLIB) // && (CIMG_LINK_PICOJSON)
#ifndef CIMG_FACE_API_KEY
#error(define your own Face API key path)
#endif

		////////////////////////////////////////////////////////////////////////////////// face detection api
		httplib::Client cli("cimgproc-test1.cognitiveservices.azure.com");
		std::shared_ptr<httplib::Response> response = 0;

		int try_count = 5;
		while (try_count-- > 0)
		{
			std::string jam_face_api_key = Util::fileToStr(CIMG_FACE_API_KEY);
			httplib::Headers headers({{"Ocp-Apim-Subscription-Key" , jam_face_api_key }});
			std::string body(R"({"url" : "https://raw.githubusercontent.com/Maetel/CImgProc/face_api/resources/jam.jpg"})");
				
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

		

		struct Rect
		{
		public:
			Rect() = default;
			~Rect() {}

		public:
			int left = -1;
			int top = -1;
			int wid = -1;
			int hi = -1;

		public:
			
			Rect& setCoord(int left, int top, int wid, int hi)
			{
				this->left = left;
				this->top = top;
				this->wid = wid;
				this->hi = hi;
				return *this;
			}

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

		Rect face;

		//this function's not ready for exception handling
		auto parseAzureFaceJson = [](std::string const& input)->std::tuple<int, int, int, int>
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
		};

		auto result = parseAzureFaceJson(res_body);
		face.setCoord(
			std::get<0>(result),
			std::get<1>(result),
			std::get<2>(result),
			std::get<3>(result)
		);

		std::cout << face.toString() << std::endl;
		
		////////////////////////////////////////////////////////////////////////////////// end of using face detection API

		const std::string path = "jam.jpg";
		uint8_t* bgrImage = 0, *gray = 0;
		int wid, hi;
		if (!loadBGRandMakeGray(path, bgrImage, gray, wid, hi))
		{
			std::cout << "No image or image path not correct" << std::endl;
			return;
		}
		const int pxCount = wid * hi;

		cv::Mat bgr2CV(wid, hi, CV_8UC3, bgrImage);
		cv::Mat faceDetected(wid, hi, CV_8UC3);
		
		const auto faceCenterPoint = face.centerPoint();
		const auto faceRectDistance = face.farthest() * 2.;

		Rect imageRect;
		imageRect.setCoord(0, 0, wid, hi);
		const auto totalDistance = face.farthest() + imageRect.farthest();

		auto fillMask = [](int wid, int hi, Point const& centerPoint, double dist)->cv::Mat
		{
			cv::Mat retval(wid, hi, CV_32F);
			retval.setTo(0);

			for (int y = 0; y < hi; ++y)
			{
				for (int x = 0; x < wid; ++x)
				{
					const Point curPoint(x, y);
#if 1
					const auto curDist = curPoint.distanceFrom(centerPoint);
					if (curDist > dist)
						continue;

					retval.at<float>(y, x) = float(1 - curDist / dist);
#else
					if (face.containsPoint(curPoint))
						jam_mask.at<float>(y, x) = 1;
#endif
				}
			}

			return retval;
		};

		const auto faceMask = fillMask(wid, hi, faceCenterPoint, faceRectDistance);
		const auto finalMask = fillMask(wid, hi, faceCenterPoint, totalDistance);

		for(int y = 0; y < hi; ++y)
			for(int x = 0; x < wid; ++x)
				faceDetected.at<cv::Vec3b>(y,x) = faceMask.at<float>(y,x) * bgr2CV.at<cv::Vec3b>(y, x);

		cv::imwrite("faceDetected.jpg", faceDetected);

		//median for diffuse
		uint8_t* medianed = new uint8_t[pxCount * 3];
		//try loading premaid
		int wid_med, hi_med;
		bool makeMedian = true;
		if (loadImage<3>("medianedImg.jpg", medianed, wid_med, hi_med))
		{
			if (wid_med == wid && hi_med == hi)
			{
				std::cout << "premaid \"medianedImg.jpg\" loaded" << std::endl;
				makeMedian = false;
			}
		}
		
		if(makeMedian)
		{
			//if medianed image does not exist
			constexpr int kernelSize = 15;
			{
				//median for diffuse
				Util::SCOPED_TIMER(Median for diffuse);
				CIMGPROC::ImageAlg::medianFilter_t<3>(bgrImage, medianed, wid, hi, kernelSize);
			}

			if (1)
			{
				//save for later use
				cv::Mat medianedImg(hi, wid, CV_8UC3, medianed);
				cv::imwrite("medianedImg.jpg", medianedImg);
			}
		}

		//diffuse
		uint8_t* diffusedFace = new uint8_t[pxCount * 3];
		uint8_t* diffusedFinal = new uint8_t[pxCount * 3];
		uint8_t* zeros = new uint8_t[pxCount * 3];
		for (int idx = 0; idx < pxCount; ++idx) zeros[idx] = 0;

		//diffuse face
		CIMGPROC::ImageAlg::diffuse<3>(medianed, bgrImage, (float const*)faceMask.data, diffusedFace, wid * hi);

		//diffuse total
		CIMGPROC::ImageAlg::diffuse<3>(zeros, diffusedFace, (float const*)finalMask.data, diffusedFinal, wid * hi);

		if (1)
		{
			//check output
			cv::Mat diffusedFaceImg(hi, wid, CV_8UC3, diffusedFace);
			cv::imwrite("diffusedFace.jpg", diffusedFaceImg);

			cv::Mat diffusedFinalImg(hi, wid, CV_8UC3, diffusedFinal);
			cv::imwrite("finalOutput.jpg", diffusedFinalImg);
		}


		{
			//clean up
			delete[] bgrImage;
			delete[] gray;
			delete[] medianed;
			delete[] diffusedFace;
			delete[] diffusedFinal;
			delete[] zeros;
		}
#else
		std::cerr << "Library not linked. Returning..." << std::endl;
		return;
#endif
	} // ! faceDetectionAndManipulation
	
}