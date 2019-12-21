//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#include "JamExecution.h"
#include "BrettExecution.h"

int main(int argc, char* argv[])
{
	CIMGPROC::Jam jam;
	
	//jam.loadImageTester();
	//jam.convert2Gray();
	jam.histogram();

	//jam.execute();
	//jam.runCL();
	//jam.runHttpsClient();
	//jam.colorMagnet();

	return 0;
}