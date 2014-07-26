/*
	AI Lib
	Copyright (C) 2014 Eric Laukien

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

	This version has been modified from the original.
*/

#pragma once

#include <erl/simulation/Experiment.h>

class ExperimentOR : public erl::Experiment {
public:
	// Inherited from Experiment
	float evaluate(erl::Field2DGenes &fieldGenes, const neat::EvolverSettings &settings,
		const std::shared_ptr<cl::Image2D> &randomImage,
		const std::shared_ptr<cl::Program> &blurProgram,
		const std::shared_ptr<cl::Kernel> &blurKernelX,
		const std::shared_ptr<cl::Kernel> &blurKernelY,
		const std::vector<std::function<float(float)>> &activationFunctions,
		const std::vector<std::string> &activationFunctionNames,
		float minInitRec, float maxInitRec, erl::Logger &logger,
		erl::ComputeSystem &cs, std::mt19937 &generator);
};