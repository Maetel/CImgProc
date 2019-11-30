//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////
#include "CLInstance.h"
//CLInstance
namespace CIMGPROC::CL
{
	class _CLInstance::Internal
	{
	public:
		Internal() {}
		~Internal() {}

		void init()
		{
			m_platform = cl::Platform::getDefault();
			m_device = cl::Device::getDefault();
			m_context = cl::Context::getDefault();
		}

	public:
		cl::Context m_context;
		cl::Device m_device;
		cl::Platform m_platform;
	};

	_CLInstance::_CLInstance()
		: pImpl(new Internal)
	{
		pImpl->init();
	}
	_CLInstance::~_CLInstance()
	{
		delete pImpl;
		pImpl = nullptr;
	}
	cl::Context _CLInstance::context()
	{
		return _CLInstance::instance()->pImpl->m_context;
	}
	cl::Platform _CLInstance::platform() const
	{
		return _CLInstance::instance()->pImpl->m_platform;
	}
	cl::Device _CLInstance::device() const
	{
		return _CLInstance::instance()->pImpl->m_device;
	}
	std::string _CLInstance::deviceInfo() const
	{
		std::string device_info="", device_buf;
		
		device().getInfo(CL_DEVICE_NAME, &device_buf);
		device_info += device_buf + "\n";

		this->device().getInfo(CL_DEVICE_VENDOR, &device_buf);
		device_info += device_buf + "\n";

		this->device().getInfo(CL_DEVICE_VERSION, &device_buf);
		device_info += device_buf + "\n";

		return device_info;
	}
	std::string _CLInstance::platformInfo() const
	{
		std::string info = "", buf;

		platform().getInfo(CL_PLATFORM_NAME, &buf);
		info += buf + "\n";

		this->device().getInfo(CL_PLATFORM_PROFILE, &buf);
		info += buf + "\n";

		this->device().getInfo(CL_PLATFORM_VENDOR, &buf);
		info += buf + "\n";

		this->device().getInfo(CL_PLATFORM_VERSION, &buf);
		info += buf + "\n";

		return info;
	}
}
//!CLInstance