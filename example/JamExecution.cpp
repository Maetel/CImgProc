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

//std
#include <functional>

//////3rd party
//opencv
#include "opencv2/highgui.hpp"
//eigen
#include "eigen.h"
//HttpsLib
#include "httplib.h"

namespace CIMGPROC
{
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
				//ImageAlg::binarize<ImageAlg::THRESHOLD>(lenaGray.data, lena_otsu.data, pxCount, &binThres);

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

	//test MS Azure Face API
	void Jam::runHttpsClient()
	{
#ifdef CIMG_LINK_HTTPLIB

		enum {HTTP = 80, HTTPS = 443};
		httplib::Client cli(
			"cimgproc-test1.cognitiveservices.azure.com",
			HTTPS,
			5
		);

		int try_count = 5;

		while (try_count-- > 0)
		{
			httplib::Params items;
			items.emplace("url", "https://github.com/Maetel/CImgProc/blob/master/resources/lena.jpg?raw=true");

#ifndef CIMG_FACE_API_KEY
#define CIMG_FACE_API_KEY error(define your own Face API key path)
#endif
			std::string jam_face_api_key = Util::fileToStr(CIMG_FACE_API_KEY);

			httplib::Headers headers({
					{"Accept", "application/json, text/plain, */*"},
					{"Content-Type", "application/json;charset=utf-8"},
					{"Ocp-Apim-Subscription-Key" , jam_face_api_key }
				});

			std::string body(R"({"url" : "https://github.com/Maetel/CImgProc/blob/master/resources/lena.jpg?raw=true"})");

			auto res = cli.Post(
				"https://cimgproc-test1.cognitiveservices.azure.com/vision/v2.0/analyze?visualFeatures=Faces&details=Landmarks&language=en",
				headers,
				body,
				"application/json;charset=utf-8"
			);

			if (res)
			{
				std::cout << res->body << std::endl;
			}
			else
			{
				std::cout << "No response : " << try_count << std::endl;;
				Sleep(1000);
				continue;
			}

		}
#endif
	} // ! runHttpsClient
}




