/*
	NEAT Visualizer
	Copyright (C) 2012-2014 Eric Laukien

	This software is provided 'as-is', without any express or implied
	warranty.  In no event will the authors be held liable for any damages
	arising from the use of this software.

	Permission is granted to anyone to use this software for any purpose,
	including commercial applications, and to alter it and redistribute it
	freely, subject to the following restrictions:

	1. The origin of this software must not be misrepresented; you must not
		claim that you wrote the original software. If you use this software
		in a product, an acknowledgment in the product documentation would be
		appreciated but is not required.
	2. Altered source versions must be plainly marked as such, and must not be
		misrepresented as being the original software.
	3. This notice may not be removed or altered from any source distribution.

	This version of the NEAT Visualizer has been modified for ERL to include different activation functions (CPPN)
*/

#pragma once

#include <stdint.h>
#include <string>

// Override macro, to remain cross platform. Override keyword is only available in VS. If not using VS, OVERRIDE does nothing
#ifdef _MSC_BUILD
#define OVERRIDE override
#else
#define OVERRIDE
#endif

#define _USE_MATH_DEFINES
#include <math.h>

namespace neat {
	template<class T>
	T wrap(T val, T size) {
		if (val < 0)
			return val + size;

		if (val >= size)
			return val - size;

		return val;
	}

	inline float sign(float val) {
		return val < 0.0f ? -1.0f : 1.0f;
	}
}