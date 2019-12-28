//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#if 1
#include "CLAlgExample.h"
#include "CLConvolution.h"
#include "CLConvolution.cpp"

namespace CIMGPROC
{
	void CLImageAlgExample::runCL()
	{
		auto context = CLInstance.context();
		auto queue = cl::CommandQueue(context);
		
		CL::CLConvolution convolution;
		convolution.build();
		
	}
}
#else
#include "CLAlgExample.h"
namespace CIMGPROC
{
	void CLImageAlgExample::runCL()
	{
	}
}
#endif