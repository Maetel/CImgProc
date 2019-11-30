//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#ifndef CIMGPROC_CL_RGB2GRAY_H
#define CIMGPROC_CL_RGB2GRAY_H

#include "CLAlgorithm.h"
namespace CIMGPROC::CL
{
	class CLConvert2Gray : public CLAlgorithm
	{
	public:
		enum Convert2Gray { RGB2GRAY = 0, BGR2GRAY, RGBA2GRAY, BGRA2GRAY, };
		CLConvert2Gray();
		~CLConvert2Gray();

	public:
		void rebuildProgram(std::string const& option) override;
		bool convert2gray(cl::CommandQueue& queue, cl::Buffer const& input, cl::Buffer& output, int length, Convert2Gray cvtType);

	protected:
		cl::Program program;
	};
}
#endif //!CIMGPROC_CL_RGB2GRAY_H
