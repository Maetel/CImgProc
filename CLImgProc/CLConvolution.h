//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#ifndef CIMGPROC_CL_CONVOLUTION_H
#define CIMGPROC_CL_CONVOLUTION_H

#include "CLAlgorithm.h"
#include <map>
namespace CIMGPROC::CL
{
	struct KernelBuffer
	{
		KernelBuffer() {}
		KernelBuffer(cl::Context const& context, float const* kernel, int kern_wid, int kern_hi, bool normalize)
		{
			m_buffer = kernelToBuffer(context, kernel, kern_wid, kern_hi, normalize);
		}
		~KernelBuffer() {}

		cl::Buffer kernelToBuffer(cl::Context const& context, float const* kernel, int kern_wid, int kern_hi, bool normalize)
		{
			m_normalize = normalize;
			m_kern_wid = kern_wid;
			m_kern_hi = kern_hi;
			auto queue = cl::CommandQueue(context);
			const size_t size = sizeof(float) * kern_wid * kern_hi;
			m_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, size);
			queue.enqueueWriteBuffer(m_buffer, true, 0, size, kernel);
			queue.flush();
			queue.finish();

			return m_buffer;
		}

		bool m_normalize = true;
		int m_kern_wid = -1, m_kern_hi = -1;
		cl::Buffer m_buffer;
	};

	class CLConvolution : public CLAlgorithm
	{
	public:
		CLConvolution();
		~CLConvolution();

	public:
		void rebuildProgram() override;

		bool convolution(cl::CommandQueue& queue,
			cl::Buffer const& input, cl::Buffer& output,
			int wid, int hi,
			KernelBuffer const& kernel
		);

		bool convolution(cl::CommandQueue& queue,
			cl::Buffer const& input, cl::Buffer& output,
			int wid, int hi,
			cl::Buffer const& kernel,
			int kern_wid, int kern_hi,
			bool normalize
			);

		bool sobel_dx(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi);
		bool sobel_dy(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi);
		bool scharr_dx(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi);
		bool scharr_dy(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi);
		bool sharpen(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi);
		bool sharpen_laplacian(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi);
		//last created gaussian kernel will be cached internally
		bool gaussianConvolution(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int wid, int hi, int gaussKernSize, double gaussSigma);

	protected:
		std::map<std::string, cl::Program> m_programs;
		class Internal;
		Internal* pImpl;
	};
	using CLGaussian = CLConvolution;
	using CLSobel = CLConvolution;
	using CLScharr = CLConvolution;
}
#endif //!CIMGPROC_CL_CONVOLUTION_H
