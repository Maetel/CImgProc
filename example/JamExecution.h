//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#ifndef CIMGPROC_JAM_EXECUTION_H
#define CIMGPROC_JAM_EXECUTION_H

namespace CIMGPROC
{
	class Jam
	{
	public:
		void execute();
		void runCL();
		void runHttpsClient();
		void colorMagnet();
	};
}


#endif //!CIMGPROC_JAM_EXECUTION_H