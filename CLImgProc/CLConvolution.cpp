//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#include "CLConvolution.h"
#include "JamUtility.h"

#ifndef CIMGPROC_MATH_H
namespace CIMGPROC
{
	constexpr double PI = 3.1415926535897932;
}
#endif // !PI
#ifndef CIMGPROC_IMAGEALG_H
namespace CIMGPROC::CL
{
	//https://www.geeksforgeeks.org/gaussian-filter-generation-c/
	template <typename T = double>
	inline void gaussianKernelGeneration(T * output, int kernelSize = 3, double sigma = 1.0)
	{
		// intialising standard deviation
		double s = 2.0 * sigma * sigma;

		// sum is for normalization 
		double sum = 0.0;

		const int kernOffset = kernelSize / 2;

		// generating kernel 
		for (int x = -kernOffset; x <= kernOffset; x++) {
			for (int y = -kernOffset; y <= kernOffset; y++) {
				const int dst = x + kernOffset + (y + kernOffset) * kernelSize;
				output[dst] = (std::exp(-1 * (x * x + y * y) / s)) / (CIMGPROC::PI * s);
				sum += output[dst];
			}
		}

		// normalizing the Kernel 
		for (int i = 0; i < kernelSize; ++i)
			for (int j = 0; j < kernelSize; ++j)
				output[i + (j * kernelSize)] /= sum;
	} //!gaussianKernelGeneration
}
#endif // !gaussianKernelGeneration

namespace CIMGPROC::CL::CONST_KERNEL
{
	constexpr float _CL_SOBEL_DX[] =
	{
		 3,	0,  -3,
		10,	0, -10,
		 3,	0,  -3
	};
	constexpr float _CL_SOBEL_DY[] =
	{
		 3,	 10,  3,
		 0,	  0,  0,
		-3,	-10, -3
	};
	constexpr float _CL_SCHARR_DX[] =
	{
		47,	 0, -47,
		162, 0, -162,
		47,	 0, -47
	};
	constexpr float _CL_SCHARR_DY[] =
	{
		 47,  162,  47,
		  0,    0,   0,
		-47, -162, -47
	};
	constexpr float _CL_SHARPEN[] =
	{
		-1, -1, -1,
		-1,  9, -1,
		-1, -1, -1
	};
	constexpr float _CL_SHARPEN_LAPLACIAN[] =
	{
		0,  1, 0,
		1, -4, 1,
		0,  1, 0
	};
}

namespace CIMGPROC::CL
{
	CL::KernelBuffer gaussianBuffer(int gaussLength, double sigma)
	{
		std::vector<float> gaussian(gaussLength * gaussLength);
		gaussianKernelGeneration(gaussian.data(), gaussLength, sigma);
		return CL::KernelBuffer(CLInstance.context(), gaussian.data(), gaussLength, gaussLength, true);
	}
}

namespace CIMGPROC::CL
{
	class CLConvolution::Internal
	{
	public:
		enum ConvoType {
			SOBEL_DX = 0, SOBEL_DY,
			SCHARR_DX, SCHARR_DY,
			SHARPEN, SHARPEN_LAPLACIAN,
			GAUSSIAN,
			MAX_NUM
		};
	public:
		Internal()
			: m_kernelBuffers(MAX_NUM)
		{
			makeKernelBuffers();
		}
		~Internal() {}

	public:
		CL::KernelBuffer kernelBuffers(ConvoType type)
		{
			return m_kernelBuffers.at(type);
		}
		void makeKernelBuffers()
		{
			using namespace CONST_KERNEL;

			m_kernelBuffers = std::vector<CL::KernelBuffer>(
				{
					CL::KernelBuffer(CLInstance.context(), _CL_SOBEL_DX, 3, 3, false),
					CL::KernelBuffer(CLInstance.context(), _CL_SOBEL_DY, 3, 3, false),
					CL::KernelBuffer(CLInstance.context(), _CL_SCHARR_DX, 3, 3, false),
					CL::KernelBuffer(CLInstance.context(), _CL_SCHARR_DX, 3, 3, false),
					CL::KernelBuffer(CLInstance.context(), _CL_SHARPEN, 3, 3, true),
					CL::KernelBuffer(CLInstance.context(), _CL_SHARPEN_LAPLACIAN, 3, 3, true),
					gaussianBuffer(m_last_gauss_length, m_last_gauss_sigma)
				}
			);
			
		}

	public:
		void updateGaussianBuffer(int gaussKernLength, double sigma)
		{
			if (gaussKernLength == m_last_gauss_length && std::abs(sigma - m_last_gauss_sigma) < 0.0001)
				return;
			m_last_gauss_length = gaussKernLength;
			m_last_gauss_sigma = sigma;
			m_kernelBuffers.at(GAUSSIAN) = gaussianBuffer(m_last_gauss_length, m_last_gauss_sigma);
		}

	protected:
		

	protected:
		std::vector<CL::KernelBuffer> m_kernelBuffers;
		CL::KernelBuffer m_gaussBuffer;
		int m_last_gauss_length = 3;
		double m_last_gauss_sigma = 1.0;
	};

	CLConvolution::CLConvolution()
		: pImpl(new Internal)
	{
	}

	CLConvolution::~CLConvolution()
	{
		delete pImpl;
		pImpl = nullptr;
	}

	std::string toProgramMap(int x, int y)
	{
		return std::to_string(x) + "*" + std::to_string(y);
	}

	void CLConvolution::rebuildProgram()
	{
		//program = this->buildProgram("convert2gray.cl", option);

		//m_programs["x*y"]
		for (int y = 3; y < 20; y+=2)
		{
			for (int x = 3; x < 20; x+=2)
			{
				const std::string mapDst = toProgramMap(x, y);
				const std::string s(
#include "cl_srcs/convolution.cl"
				);
				m_programs[mapDst] = buildProgramWithSrc(s, kernSizeDefinition(x, y));
			}
		}
	}

	bool CLConvolution::convolution(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi, KernelBuffer const& kernel)
	{
		return convolution(queue, input, output, wid, hi, kernel.m_buffer, kernel.m_kern_wid, kernel.m_kern_hi, kernel.m_normalize);
	}

	bool CLConvolution::convolution(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi, cl::Buffer const& kernel, int kern_wid, int kern_hi, bool normalize)
	{
		if (nullptr == queue.get() || nullptr == input.get() || nullptr == output.get() || wid <= 0 || hi <= 0 || nullptr == kernel.get())
			return false;

		const auto mapDst = toProgramMap(kern_wid, kern_hi);
		auto prgm = m_programs.find(mapDst);
		if (prgm == m_programs.end())
		{
			//not supported kernel size
			return false;
		}
			
		auto _kernel = cl::Kernel((*prgm).second, "convolution");

		cl::EnqueueArgs args(
			queue,
			roundUpGlobalSize(wid, hi),
			localSize()
		);

		const int _normalize = !!normalize;

		cl::KernelFunctor<
			cl::Buffer, //input
			cl::Buffer, //kernel
			cl::Buffer,	//output
			int, int,	//wid, hi
			int			//normalize
		> functor(_kernel);

		cl_int err;
		functor(args,
			input,
			kernel,
			output,
			wid, hi,
			_normalize,
			err
		);

		return (CL_SUCCESS == err);
	}

	bool CLConvolution::sobel_dx(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi)
	{
		return convolution(queue, input, output, wid, hi, pImpl->kernelBuffers(Internal::SOBEL_DX));
	}

	bool CLConvolution::sobel_dy(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi)
	{
		return convolution(queue, input, output, wid, hi, pImpl->kernelBuffers(Internal::SOBEL_DY));
	}

	bool CLConvolution::scharr_dx(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi)
	{
		return convolution(queue, input, output, wid, hi, pImpl->kernelBuffers(Internal::SCHARR_DX));
	}

	bool CLConvolution::scharr_dy(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi)
	{
		return convolution(queue, input, output, wid, hi, pImpl->kernelBuffers(Internal::SCHARR_DY));
	}

	bool CLConvolution::sharpen(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi)
	{
		return convolution(queue, input, output, wid, hi, pImpl->kernelBuffers(Internal::SHARPEN));
	}

	bool CLConvolution::sharpen_laplacian(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi)
	{
		return convolution(queue, input, output, wid, hi, pImpl->kernelBuffers(Internal::SHARPEN_LAPLACIAN));
	}

	bool CLConvolution::gaussianConvolution(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi, int gaussKernSize, double gaussSigma)
	{
		pImpl->updateGaussianBuffer(gaussKernSize, gaussSigma);
		return convolution(queue, input, output, wid, hi, pImpl->kernelBuffers(Internal::GAUSSIAN));
	}

}
