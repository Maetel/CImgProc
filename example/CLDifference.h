//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#ifndef CIMGPROC_CL_DIFFERENCE_H
#define CIMGPROC_CL_DIFFERENCE_H

#include "CLAlgorithm.h"
namespace CIMGPROC::CL
{
	class CLDifference : public CLAlgorithm
	{
	public:
		CLDifference();
		~CLDifference();

	public:
		void rebuildProgram(std::string const& option) override;
		bool difference(cl::CommandQueue& queue, cl::Buffer const& input_l, cl::Buffer const& input_r, cl::Buffer& output, int length);

	protected:
		cl::Program program;
	};
}
#endif //!CIMGPROC_CL_DIFFERENCE_H
