//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////
#include "CLAlgorithm.h"
#include <fstream>
#include "JamUtility.h"

namespace CIMGPROC::CL
{
	const static std::string _include() { return R"(#include "includes.cl")"; }
	const static std::string _includes() { return Util::fileToStr(CL_SRC_DIRPATH "includes.cl"); }
	const static std::string _externOptions() { return R"(#external_options)"; }

	void CLAlgorithm::build()
	{
		this->rebuildProgram();
	}

	cl::Program CLAlgorithm::buildProgram(std::string const& fileName, std::string const& options)
	{
		auto src = Util::fileToStr(std::string(CL_SRC_DIRPATH) + fileName);
		Util::replace(src, _include(), _includes());
		Util::replace(src, _externOptions(), options);
		constexpr bool build = true;
		cl_int err;

		//if(std::string("convolution.cl") == fileName)
		//{
		//	std::cout << src << std::endl;
		//}

		cl::Program prgm(context(), src, build, &err);

		if (CL_SUCCESS != err)
		{
			std::string err_log;
			prgm.getBuildInfo(device(), CL_PROGRAM_BUILD_LOG, &err_log);
			std::cout << "[" __FUNCTION__ << ", " << __LINE__ << "]\n" << src<< "\n=====================================================================\n" << err_log << std::endl;
			return nullptr;
		}

		return prgm;
	}
	cl::NDRange CLAlgorithm::localSize() const
	{
		return m_localSize;
	}
	cl::NDRange CLAlgorithm::setLocalSize(int size1)
	{
		m_localSize = cl::NDRange(size1);
		return m_localSize;
	}
	cl::NDRange CLAlgorithm::setLocalSize(int size1, int size2)
	{
		m_localSize = cl::NDRange(size1, size2);
		return m_localSize;
	}
	cl::NDRange CLAlgorithm::setLocalSize(int size1, int size2, int size3)
	{
		m_localSize = cl::NDRange(size1, size2, size3);
		return m_localSize;
	}

	size_t roundUp(int dst, int divider)
	{
		const auto rest = dst % divider;
		return (0 == rest) ? dst : (dst + divider - rest);
	}

	cl::NDRange CLAlgorithm::roundUpGlobalSize(int _1dSize) const
	{
		return cl::NDRange(roundUp(_1dSize, *m_localSize));
	}
	cl::NDRange CLAlgorithm::roundUpGlobalSize(int _2dSize1, int _2dSize2) const
	{
		return cl::NDRange(roundUp(_2dSize1, *m_localSize), roundUp(_2dSize2, *m_localSize));
	}
	cl::NDRange CLAlgorithm::roundUpGlobalSize(int _3dSize1, int _3dSize2, int _3dSize3) const
	{
		return cl::NDRange(roundUp(_3dSize1, *m_localSize), roundUp(_3dSize2, *m_localSize), roundUp(_3dSize3, *m_localSize));
	}
	std::string CLAlgorithm::kernSizeDefinition(int x, int y) const
	{
		std::string kernSizeOption(R"(
#define KERN_WID __KERN_WID
#define KERN_HI __KERN_HI
)");
		Util::replace(kernSizeOption, "__KERN_WID", std::to_string(x));
		Util::replace(kernSizeOption, "__KERN_HI", std::to_string(y));
		return kernSizeOption;
	}
}
