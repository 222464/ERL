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

#include <neat/ConnectionGene.h>
#include <vector>
#include <random>

namespace neat {
	class Evolvable {
	private:
	public:
		virtual ~Evolvable() {}

		virtual void initialize(size_t numInputs, size_t numOutputs, const class EvolverSettings* pSettings, const std::vector<float> &functionChances, InnovationNumberType &innovationNumber, std::mt19937 &generator) = 0;
		virtual void crossover(const class EvolverSettings* pSettings, const std::vector<float> &functionChances, const Evolvable* pOtherParent, Evolvable* pChild, float fitnessForThis, float fitnessForOtherParent, InnovationNumberType &innovationNumber, std::mt19937 &generator) = 0;
		virtual void mutate(const class EvolverSettings* pSettings, const std::vector<float> &functionChances, InnovationNumberType &innovationNumber, std::mt19937 &generator) = 0;
		virtual float getSimilarity(const class EvolverSettings* pSettings, const std::vector<float> &functionChances, const Evolvable* pOther) { return 0.0f; }
	};
}