//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#ifndef CIMGPROC_CL_DIFFUSE_H
#define CIMGPROC_CL_DIFFUSE_H

#include "CLAlgorithm.h"
namespace CIMGPROC::CL
{
	class CLDiffuse : public CLAlgorithm
	{
	public:
		CLDiffuse();
		~CLDiffuse();

	public:
		void rebuildProgram() override;

		//diffuse with ratio
		bool diffuse(cl::CommandQueue& queue, cl::Buffer const& input1, cl::Buffer const& input2, cl::Buffer& output, float ratio_1to2, int length);

		//diffuse with mask
		bool diffuse(cl::CommandQueue& queue, cl::Buffer const& input1, cl::Buffer const& input2, cl::Buffer& output, cl::Buffer const& mask, int length);

	protected:
		cl::Program program;
	};
}
#endif //!CIMGPROC_CL_DIFFUSE_H
