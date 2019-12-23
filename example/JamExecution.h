//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////

#ifndef CIMGPROC_JAM_EXECUTION_H
#define CIMGPROC_JAM_EXECUTION_H

#include <string>
#include <utility>

namespace CIMGPROC
{
	class Jam
	{
		using uint8_t = unsigned char;
	public:
		template <int Channel = 3, typename ...Params>
		bool loadImage(Params&&... params)
		{
#define LOAD_N_CHANNEL_IMAGE(_ch) \
if constexpr (_ch == Channel) return _loadImage##_ch(std::forward<Params>(params)...); else
			LOAD_N_CHANNEL_IMAGE(1)
			LOAD_N_CHANNEL_IMAGE(2)
			LOAD_N_CHANNEL_IMAGE(3)
			LOAD_N_CHANNEL_IMAGE(4)
				return false;
#undef LOAD_N_CHANNEL_IMAGE
		}

#define DECL_IMPL_LOADIMAGE(_ch) \
bool _loadImage##_ch(std::string const& path, uint8_t* &data, int& wid, int& hi);
		DECL_IMPL_LOADIMAGE(1);
		DECL_IMPL_LOADIMAGE(2);
		DECL_IMPL_LOADIMAGE(3);
		DECL_IMPL_LOADIMAGE(4);
#undef DECL_IMPL_LOADIMAGE

		bool loadBGRandMakeGray(std::string const& path, uint8_t*& bgr, uint8_t*& gray, int& wid, int& hi);

		void loadImageTester();
		void convert2Gray();
		void histogram();
		void NCC();
		void ZNCC();
		void binarization();
		void gaussian();
		void derivation();
		void convolution();
		void colorMagnet();

		void execute();
		void runCL();
		void runHttpsClient();
	};
}


#endif //!CIMGPROC_JAM_EXECUTION_H