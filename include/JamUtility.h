//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////
#ifndef CIMGPROC_UTILITY_H
#define CIMGPROC_UTILITY_H

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

namespace CIMGPROC::Util
{

	//https://stackoverflow.com/questions/3418231/replace-part-of-a-string-with-another-string
	inline bool replace(std::string& str, const std::string& from, const std::string& to) {
		size_t start_pos = str.find(from);
		if (start_pos == std::string::npos)
			return false;
		str.replace(start_pos, from.length(), to);
		return true;
	}

	template<typename Text>
	inline std::string fileToStr(Text const& filePath)
	{
		std::ifstream stream(filePath);
		return std::string(
			(std::istreambuf_iterator<char>(stream)),
			std::istreambuf_iterator<char>()
		);
	}

	class __ScopedTimer
	{
		using Timer = std::chrono::high_resolution_clock;
	public:
		__ScopedTimer(std::string const& text) : m_text(text) { m_timer = Timer::now(); }
		~__ScopedTimer()
		{
			auto finished = Timer::now();
			const auto elapsed_usec = std::chrono::duration_cast<std::chrono::microseconds>(finished - m_timer).count();
			const bool isOver3ms = elapsed_usec > 3000;
			const bool isOver2s = elapsed_usec > 2000000;
			std::cout << m_text << "  " << (isOver2s ? (elapsed_usec / 1000000) : (isOver3ms ? (elapsed_usec / 1000) : elapsed_usec)) << (isOver2s ? "s" : (isOver3ms ? "ms" : "us")) << std::endl;
		}

	private:
		std::string m_text;
		std::chrono::steady_clock::time_point m_timer;
	};
#define SCOPED_TIMER(x) __ScopedTimer __scopedTimer##__LINE__(std::string("[Timer @" __FUNCTION__ " @") + std::to_string(__LINE__) + std::string("] " #x));
}

#endif //!CIMGPROC_UTILITY_H
