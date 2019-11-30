//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#ifndef CIMGPROC_CL_INSTANCE_H
#define CIMGPROC_CL_INSTANCE_H

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "CL/cl2.hpp"

#ifndef CL_SRC_DIRPATH
#define CL_SRC_DIRPATH "variable \"CL_SRC_DIRPATH\" not set"
#endif

namespace CIMGPROC::CL
{
	class _CLInstance
	{
	public:
		_CLInstance();
		~_CLInstance();
		
	public:
		static _CLInstance* instance()
		{
			static _CLInstance instance;
			return &instance;
		}
		
	public:
		cl::Context context();
		cl::Platform platform() const;
		cl::Device device() const;
		std::string platformInfo() const;
		std::string deviceInfo() const;

	protected:
		class Internal;
		Internal* pImpl = 0;
	};
#define CLInstance (*::CIMGPROC::CL::_CLInstance::instance())
}
#endif //!CIMGPROC_CL_INSTANCE_H