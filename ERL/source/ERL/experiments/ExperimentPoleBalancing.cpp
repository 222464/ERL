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

#include <erl/experiments/ExperimentPoleBalancing.h>

#include <SFML/System.hpp>

#include <iostream>

float ExperimentPoleBalancing::evaluate(erl::Field2DGenes &fieldGenes, const neat::EvolverSettings &settings,
	const std::shared_ptr<cl::Image2D> &randomImage,
	const std::shared_ptr<cl::Program> &blurProgram,
	const std::shared_ptr<cl::Kernel> &blurKernelX,
	const std::shared_ptr<cl::Kernel> &blurKernelY,
	const std::vector<std::function<float(float)>> &activationFunctions,
	const std::vector<std::string> &activationFunctionNames,
	float minInitRec, float maxInitRec, erl::Logger &logger,
	erl::ComputeSystem &cs, std::mt19937 &generator)
{
	erl::Field2DCL field;

	field.create(fieldGenes, cs, 32, 32, 2, 4, 1, 1, 1, randomImage, blurProgram, blurKernelX, blurKernelY, activationFunctions, activationFunctionNames, minInitRec, maxInitRec, generator, logger);

	std::uniform_real_distribution<float> initPosDist(-1.0f, 1.0f);
	std::uniform_real_distribution<float> initPoleVelDist(-0.05f, 0.05f);

	float pixelsPerMeter = 128.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 20.0f;
	float cartMass = 2.0f;
	sf::Vector2f massVel(0.0f, 0.0f);
	float poleAngle = static_cast<float>(std::_Pi) * 0.0f;
	float poleAngleVel = initPoleVelDist(generator);
	float poleAngleAccel = 0.0f;
	float cartX = initPosDist(generator);
	sf::Vector2f massPos(cartX, poleLength);
	float cartVelX = 0.0f;
	float cartAccelX = 0.0f;
	float poleRotationalFriction = 0.008f;
	float cartMoveRadius = 1.8f;
	float cartFriction = 0.02f;
	float maxSpeed = 3.0f;

	float dt = 0.03f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	float totalFitness = 0.0f;

	for (size_t i = 0; i < 600; i++) {
		//std::cout << "Step " << i << std::endl;

		// Update fitness
		prevFitness = fitness;

		if (poleAngle < static_cast<float>(std::_Pi))
			fitness = -(static_cast<float>(std::_Pi) * 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(std::_Pi) * 0.5f - (static_cast<float>(std::_Pi) * 2.0f - poleAngle));

		//fitness = fitness - std::fabsf(poleAngleVel * 40.0f);

		totalFitness += fitness * 0.1f;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		float error = dFitness * 10.0f;

		//agent.reinforceArp(std::min(1.0f, std::max(-1.0f, error)) * 0.5f + 0.5f, 0.1f, 0.05f);

		field.setInput(0, cartX * 0.25f);
		field.setInput(1, cartVelX);
		field.setInput(2, std::fmodf(poleAngle + static_cast<float>(std::_Pi), 2.0f * static_cast<float>(std::_Pi)));
		field.setInput(3, poleAngleVel);

		field.update(error, cs, activationFunctions, 16, generator);

		float dir = std::min<float>(1.0f, std::max<float>(-1.0f, field.getOutput(0)));

		//dir = 1.4f * (dir * 2.0f - 1.0f);

		float agentForce = 4000.0f * dir;
		//float agentForce = 2000.0f * agent.getOutput(0);

		// ---------------------------- Physics ----------------------------

		float pendulumCartAccelX = cartAccelX;

		if (cartX < -cartMoveRadius)
			pendulumCartAccelX = 0.0f;
		else if (cartX > cartMoveRadius)
			pendulumCartAccelX = 0.0f;

		poleAngleAccel = pendulumCartAccelX * std::cosf(poleAngle) + g * std::sinf(poleAngle);
		poleAngleVel += -poleRotationalFriction * poleAngleVel + poleAngleAccel * dt;
		poleAngle += poleAngleVel * dt;

		massPos = sf::Vector2f(cartX + std::cosf(poleAngle + static_cast<float>(std::_Pi) * 0.5f) * poleLength, std::sinf(poleAngle + static_cast<float>(std::_Pi) * 0.5f) * poleLength);

		float force = 0.0f;

		if (std::fabsf(cartVelX) < maxSpeed)
			force = std::max<float>(-4000.0f, std::min<float>(4000.0f, agentForce));

		if (cartX < -cartMoveRadius) {
			cartX = -cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}
		else if (cartX > cartMoveRadius) {
			cartX = cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}

		cartAccelX = 0.25f * (force + massMass * poleLength * poleAngleAccel * std::cosf(poleAngle) - massMass * poleLength * poleAngleVel * poleAngleVel * std::sinf(poleAngle)) / (massMass + cartMass);
		cartVelX += -cartFriction * cartVelX + cartAccelX * dt;
		cartX += cartVelX * dt;

		poleAngle = std::fmodf(poleAngle, (2.0f * static_cast<float>(std::_Pi)));

		if (poleAngle < 0.0f)
			poleAngle += static_cast<float>(std::_Pi) * 2.0f;

		cl::flush();
	}

	std::cout << "Pole balancing experiment finished with fitness of " << totalFitness << "." << std::endl;

	return totalFitness;
}