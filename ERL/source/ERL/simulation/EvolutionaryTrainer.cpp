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
*/

#include <erl/simulation/EvolutionaryTrainer.h>
#include <erl/field/Field2DGenes.h>

#include <algorithm>

using namespace erl;

EvolutionaryTrainer::EvolutionaryTrainer()
: _runsPerExperiment(10),
_numElites(3),
_greedExponent(2.0f)
{}

void EvolutionaryTrainer::create(size_t populationSize, 
	const Field2DEvolverSettings* pSettings,
	const std::vector<float> &functionChances,
	const std::shared_ptr<cl::Image2D> &randomImage,
	const std::shared_ptr<cl::Program> &blurProgram,
	const std::shared_ptr<cl::Kernel> &blurKernelX,
	const std::shared_ptr<cl::Kernel> &blurKernelY,
	const std::vector<std::function<float(float)>> &activationFunctions,
	const std::vector<std::string> &activationFunctionNames,
	float minInitRec, float maxInitRec,
	std::mt19937 &generator)
{
	_randomImage = randomImage;
	_blurProgram = blurProgram;
	_blurKernelX = blurKernelX;
	_blurKernelY = blurKernelY;
	_activationFunctions = activationFunctions;
	_activationFunctionNames = activationFunctionNames;
	_minInitRec = minInitRec;
	_maxInitRec = maxInitRec;
	_evolutionaryAlgorithm.create(populationSize, pSettings, functionChances, generator);
}

void EvolutionaryTrainer::evaluate(const Field2DEvolverSettings* pSettings,
	const std::vector<float> &functionChances, 
	ComputeSystem &cs, Logger &logger, std::mt19937 &generator)
{
	std::vector<std::vector<float>> fitnesses;

	fitnesses.resize(_experiments.size());

	for (size_t i = 0; i < _experiments.size(); i++)
		fitnesses[i].resize(_evolutionaryAlgorithm.getPopulationSize());

	for (size_t i = 0; i < _evolutionaryAlgorithm.getPopulationSize(); i++)
	for (size_t j = 0; j < _experiments.size(); j++) {
		float experimentFitness = 0.0f;

		logger << "Evaluating individual " << std::to_string(i + 1) << " of " << std::to_string(_evolutionaryAlgorithm.getPopulationSize()) << endl;

		for (size_t k = 0; k < _runsPerExperiment; k++)
			experimentFitness += _experiments[j]->evaluate(*std::static_pointer_cast<Field2DGenes>(_evolutionaryAlgorithm.getPopulationMember(i)), pSettings, _randomImage, _blurProgram, _blurKernelX, _blurKernelY, _activationFunctions, _activationFunctionNames, _minInitRec, _maxInitRec, logger, cs, generator);

		experimentFitness /= _runsPerExperiment;

		logger << "Individual " << std::to_string(i + 1) << "'s total fitness for experiment " << std::to_string(j + 1) << ": " << std::to_string(experimentFitness) << endl;

		fitnesses[j][i] = experimentFitness;
	}

	// Normalize fitness for each experiment
	for (size_t i = 0; i < _experiments.size(); i++) {
		float minimum = fitnesses[i][0];
		float maximum = fitnesses[i][0];

		for (size_t j = 0; j < fitnesses[i].size(); j++) {
			minimum = std::min<float>(minimum, fitnesses[i][j]);
			maximum = std::max<float>(maximum, fitnesses[i][j]);
		}

		if (maximum == minimum) {
			for (size_t j = 0; j < fitnesses[i].size(); j++)
				fitnesses[i][j] = 0.0f;
		}
		else {
			float rangeInv = 1.0f / (maximum - minimum);

			// Rescale fitnesses
			for (size_t j = 0; j < fitnesses[i].size(); j++)
				fitnesses[i][j] = (fitnesses[i][j] - minimum) * rangeInv;
		}
	}

	// Set fitnesses, scaled by experiment weight
	for (size_t i = 0; i < _evolutionaryAlgorithm.getPopulationSize(); i++) {
		float sum = 0.0f;

		for (size_t j = 0; j < _experiments.size(); j++)
			sum += fitnesses[j][i] * _experiments[j]->getExperimentWeight();

		_evolutionaryAlgorithm.setFitness(i, sum);
	}
}

void EvolutionaryTrainer::reproduce(const Field2DEvolverSettings* pSettings,
	const std::vector<float> &functionChances, std::mt19937 &generator)
{
	_evolutionaryAlgorithm.epoch(pSettings, functionChances, generator, _numElites, _greedExponent);
}

void EvolutionaryTrainer::writeBestToStream(std::ostream &os) const {
	size_t highestIndex = 0;

	for (size_t i = 1; i < _evolutionaryAlgorithm.getPopulationSize(); i++)
	if (_evolutionaryAlgorithm.getFitness(i) > _evolutionaryAlgorithm.getFitness(highestIndex))
		highestIndex = i;

	std::static_pointer_cast<Field2DGenes>(_evolutionaryAlgorithm.getPopulationMember(highestIndex))->writeToStream(os);
}