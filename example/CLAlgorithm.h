//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#ifndef CIMGPROC_CL_ALGORITHM_H
#define CIMGPROC_CL_ALGORITHM_H

#include "CLInstance.h"
namespace CIMGPROC::CL
{
	class CLAlgorithm
	{
	public:
		void build(std::string const& option = "");
		virtual void rebuildProgram(std::string const& option) = 0;

	public:
		cl::Context context() { return (_CLInstance::instance()->context()); }
		cl::Platform platform() { return (_CLInstance::instance()->platform()); }
		cl::Device device() { return (_CLInstance::instance()->device()); }

	protected:
		CLAlgorithm() : m_localSize(8, 8) {};
		CLAlgorithm(int size1) : m_localSize(size1) {};
		CLAlgorithm(int size1, int size2) : m_localSize(size1, size2) {};
		CLAlgorithm(int size1, int size2, int size3) : m_localSize(size1, size2, size3) {};

		cl::Program buildProgram(std::string const& filePath);
		cl::NDRange localSize() const;

		cl::NDRange setLocalSize(int);
		cl::NDRange setLocalSize(int, int);
		cl::NDRange setLocalSize(int, int, int);

		cl::NDRange roundUpGlobalSize(int) const;
		cl::NDRange roundUpGlobalSize(int, int) const;
		cl::NDRange roundUpGlobalSize(int, int, int) const;

	protected:
		cl::NDRange m_localSize;
	};

	template<typename T>
	void download(cl::CommandQueue& queue, cl::Buffer const& input, T*& output, int length, bool isBlocking = true)
	{
		if (nullptr == output)
			output = new T[length];

		queue.enqueueReadBuffer(input, isBlocking, 0, length * sizeof(T), (void*)output);
	}
}
#endif //!CIMGPROC_CL_ALGORITHM_H