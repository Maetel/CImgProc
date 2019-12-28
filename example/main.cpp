//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#include "ImageAlgExample.h"
#include "CLAlgExample.h"

int main(int argc, char* argv[])
{
	if (0)
	{
		CIMGPROC::ImageAlgExample example;
		example.loadImageTester();
		example.convert2Gray();
		example.histogram();
		example.NCC();
		example.ZNCC();
		example.binarization();
		example.gaussian();
		example.derivation();
		example.convolution();
		example.colorMagnet();
		example.differenceOfGaussian();
		example.extractChannel();
		example.median();
		example.faceDetectionAndManipulation();
	}

	if (1)
	{
		CIMGPROC::CLImageAlgExample example;
		example.runCL();
	}
	

	//example.runCL();

	return 0;
}