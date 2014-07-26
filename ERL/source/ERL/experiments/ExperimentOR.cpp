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

#include <erl/experiments/ExperimentOR.h>
#include <erl/visualization/FieldVisualizer.h>

#include <SFML/System.hpp>

#include <iostream>

float ExperimentOR::evaluate(erl::Field2DGenes &fieldGenes, const neat::EvolverSettings &settings,
	const std::shared_ptr<cl::Image2D> &randomImage,
	const std::shared_ptr<cl::Program> &blurProgram,
	const std::shared_ptr<cl::Kernel> &blurKernelX,
	const std::shared_ptr<cl::Kernel> &blurKernelY,
	const std::vector<std::function<float(float)>> &activationFunctions,
	const std::vector<std::string> &activationFunctionNames,
	float minInitRec, float maxInitRec, erl::Logger &logger,
	erl::ComputeSystem &cs, std::mt19937 &generator)
{
	float inputs[4][2] {
		{ -10.0f, -10.0f },
		{ -10.0f, 10.0f },
		{ 10.0f, -10.0f },
		{ 10.0f, 10.0f }
	};

	float outputs[4] {
		0.0f,
		1.0f,
		1.0f,
		1.0f
	};

	erl::Field2D field;

	field.create(fieldGenes, cs, 10, 10, 2, 2, 1, 3, 3, randomImage, blurProgram, blurKernelX, blurKernelY, activationFunctions, activationFunctionNames, minInitRec, maxInitRec, generator, logger);

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	std::vector<float> values(4);

	//erl::FieldVisualizer fv;
	//fv.create(cs, "adapter.cl", field, logger);

	for (size_t i = 0; i < 40; i++) {
		float newReward = 1.0f;

		float average = 0.0f;

		for (size_t j = 0; j < 4; j++) {
			field.setInput(0, inputs[j][0]);
			field.setInput(1, inputs[j][1]);

			field.update((reward - prevReward) * 10.0f, cs, activationFunctions, 14, generator);

			newReward -= std::pow(std::abs(outputs[j] - field.getOutput(0)), 2.0f) * 0.25f;

			values[j] = field.getOutput(0);
			average += values[j];
		}

		average *= 0.25f;

		//for (size_t j = 0; j < 4; j++) {
		//	newReward -= 0.05f / (0.125f + 8.0f * std::abs(average - values[j]));
		//}

		prevReward = reward;
		reward = newReward;

		if (i == 0)
			initReward = std::min<float>(1.0f, std::max<float>(0.0f, reward));

		totalReward += reward * 0.5f;

		//fv.update(cs, field);

		cl::flush();
	}

	std::cout << "Finished OR experiment with total reward of " << totalReward << "." << std::endl;

	return totalReward;
}