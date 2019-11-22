//////////////////////////////////////////////////////
// CImgProc, header-only image processing project	//
//  * Author : Wonjun Hwang							//
//  * E-mail : iamjam4944@gmail.com					//
//////////////////////////////////////////////////////
#ifndef CIMGPROC_UTILITY_H
#define CIMGPROC_UTILITY_H

#include <iostream>
#include <string>
#include <chrono>

namespace CIMGPROC {
	class ScopedTimer
	{
		using Timer = std::chrono::high_resolution_clock;
	public:
		ScopedTimer(std::string text) : m_text(text) { m_timer = Timer::now(); }
		~ScopedTimer()
		{
			auto finished = Timer::now();
			const auto elapsed_usec = std::chrono::duration_cast<std::chrono::microseconds>(finished - m_timer).count();
			const bool isOver3ms = elapsed_usec > 3000;
			std::cout << m_text << "  " << (isOver3ms ? (elapsed_usec / 1000) : elapsed_usec) << (isOver3ms ? "ms" : "us") << std::endl;
		}

	private:
		std::string m_text;
		std::chrono::steady_clock::time_point m_timer;
	};
#define SCOPED_TIMER(x) ScopedTimer __scopedTimer(std::string("[Timer @" __FUNCTION__ " @") + std::to_string(__LINE__) + std::string("] " #x));
}

#endif //!CIMGPROC_UTILITY_H
