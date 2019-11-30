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

	void CLAlgorithm::build(std::string const& option)
	{
		this->rebuildProgram(option);
	}

	cl::Program CLAlgorithm::buildProgram(std::string const& fileName)
	{
		auto src = Util::fileToStr(std::string(CL_SRC_DIRPATH) + fileName);
		Util::replace(src, _include(), _includes());
		constexpr bool build = true;
		cl_int err;

		cl::Program prgm(context(), src, build, &err);

		if (CL_SUCCESS != err)
		{
			std::string err_log;
			prgm.getBuildInfo(device(), CL_PROGRAM_BUILD_LOG, &err_log);
			std::cout << err_log << std::endl;
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
}
