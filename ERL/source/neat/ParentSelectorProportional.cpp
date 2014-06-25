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

#include <neat/ParentSelectorProportional.h>

#include <neat/Evolver.h>

#include <neat/UtilFuncs.h>

#include <algorithm>

#include <assert.h>

using namespace neat;

ParentSelectorProportional::ParentSelectorProportional()
: _numCompatibleChooseRatio(0.5f)
{}

void ParentSelectorProportional::select(EvolverSettings* pSettings, const std::vector<float> &functionChances,
	const std::vector<Evolver::GenotypeAndFitness> &pool,
	size_t &parentIndex1, size_t &parentIndex2, std::mt19937 &generator) const
{
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	const size_t poolSize = pool.size();

	assert(poolSize >= 2);

	// Roulette
	float fitnessSum = 0.0f;

	for (size_t i = 0; i < poolSize; i++)
		fitnessSum += pool[i]._fitness;

	float randomCusp = dist01(generator) * fitnessSum;

	float sumSoFar = 0.0f;

	parentIndex1 = 0;

	const size_t poolSizeOneLess = poolSize - 1;

	for (; parentIndex1 < poolSizeOneLess; parentIndex1++) {
		sumSoFar += pool[parentIndex1]._fitness;

		if (sumSoFar > randomCusp)
			break;
	}

	// Compile new list of compatible mates, and factor compatibility into the fitness measure.
	// Also keep track of most compatible if none are in the speciation tolerance
	struct IndexAndFitness {
		size_t _index;
		float _fitness;
	};

	std::vector<IndexAndFitness> compatibleGenotypes;

	fitnessSum = 0.0f;

	size_t minDisimilarityIndex = 0;
	float minDisimilarity = pool[parentIndex1]._genotype->getSimilarity(pSettings, functionChances, pool[minDisimilarityIndex]._genotype.get());

	size_t numCompatibleChoose = static_cast<size_t>(_numCompatibleChooseRatio * static_cast<float>(poolSize));

	for (size_t i = 0; i < poolSize; i++) {
		if (i == parentIndex1)
			continue;

		float disimilarity = pool[parentIndex1]._genotype->getSimilarity(pSettings, functionChances, pool[i]._genotype.get());

		if (disimilarity < minDisimilarity) {
			minDisimilarityIndex = i;
			minDisimilarity = disimilarity;
		}

		if (disimilarity < pSettings->_speciationTolerance) {
			IndexAndFitness iAF;

			iAF._index = i;

			// Weight fitness for compatibility
			fitnessSum += iAF._fitness = std::max<float>(0.0f, pool[i]._fitness - disimilarity * pSettings->_preferSimilarFactor);

			compatibleGenotypes.push_back(iAF);

			if (compatibleGenotypes.size() >= numCompatibleChoose)
				break;
		}
	}

	// If there are no compatible genes, use least disimilar
	if (compatibleGenotypes.empty()) {
		parentIndex2 = minDisimilarityIndex;

		return;
	}

	const size_t numCompatible = compatibleGenotypes.size();

	// Roulette again
	randomCusp = dist01(generator) * fitnessSum;

	sumSoFar = 0.0f;

	parentIndex2 = 0;

	for (; parentIndex2 < numCompatible; parentIndex2++) {
		if (parentIndex2 == parentIndex1)
			continue;

		sumSoFar += compatibleGenotypes[parentIndex2]._fitness;

		if (sumSoFar > randomCusp)
			break;
	}

	if (parentIndex2 == numCompatible) {
		parentIndex2--;

		if (parentIndex1 == parentIndex2)
			parentIndex2 = minDisimilarityIndex;
	}
}