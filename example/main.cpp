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
	//jam.histogram();
	//jam.NCC();
	//jam.ZNCC();
	//jam.binarization();
	//jam.gaussian();
	//jam.derivation();
	//jam.convolution();
	//jam.colorMagnet();
	//jam.differenceOfGaussian();
	//jam.extractChannel();
	//jam.median();


	//jam.execute();
	//jam.runCL();
	jam.faceDetectionAndManipulation();

	return 0;
}