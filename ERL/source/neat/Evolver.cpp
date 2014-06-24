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

#include <neat/Evolver.h>
#include <neat/ParentSelector.h>

#include <iostream>

#include <assert.h>

using namespace neat;

EvolverSettings::EvolverSettings()
: _speciationTolerance(2.2f),
_preferSimilarFactor(0.05f),
_reproduceRatio(0.9f),
_newConnectionMutationRate(0.2f),
_newNodeMutationRate(0.1f),
_weightPerturbationChance(0.2f),
_disableGeneChance(0.125f),
_minInitWeight(-1.0f),
_maxInitWeight(1.0f),
_minInitBias(-1.0f),
_maxInitBias(1.0f),
_minWeight(-8.0f),
_maxWeight(8.0f),
_minBias(-8.0f),
_maxBias(8.0f),
_maxPerturbation(0.75f),
_changeFunctionChance(0.05f),
_excessFactor(1.0f),
_disjointFactor(1.0f),
_averageWeightDifferenceFactor(0.4f),
_inputCountDifferenceFactor(1.5f),
_outputCountDifferenceFactor(1.5f),
_functionFactor(3.0f),
_populationSize(40),
_numElites(6)
{}

Evolver::Evolver()
: _innovationNumber(0)
{}

void Evolver::normalizeFitness() {
	size_t minIndex = 0;

	for (size_t i = 1; i < _population.size(); i++)
	if (_population[i]._fitness < _population[minIndex]._fitness)
		minIndex = i;

	for (size_t i = 0; i < _population.size(); i++)
		_population[i]._fitness -= _population[minIndex]._fitness;
}

void Evolver::clearPopulation() {
	_population.clear();
}

void Evolver::initialize(size_t numInputs, size_t numOutputs, const std::vector<float> &functionChances, std::shared_ptr<class ParentSelector> selector, std::mt19937 &generator, std::shared_ptr<NetworkGenotype>(*pGenotypeFactory)()) {
	clearPopulation();

	_functionChances = functionChances;

	_selector = selector;

	_pGenotypeFactory = pGenotypeFactory;

	_innovationNumber = 0;

	_population.resize(_settings._populationSize);

	// Default initialize
	for (size_t i = 0; i < _settings._populationSize; i++) {
		// Generate new gene
		std::shared_ptr<NetworkGenotype> newGenotype = _pGenotypeFactory();

		newGenotype->initialize(numInputs, numOutputs, _settings._minInitWeight, _settings._maxInitWeight, _settings._minInitBias, _settings._maxInitBias, functionChances, _innovationNumber, generator);
		newGenotype->initializeAdditional(numInputs, numOutputs, _settings._minInitWeight, _settings._maxInitWeight, _settings._minInitBias, _settings._maxInitBias, _innovationNumber, generator);

		_population[i]._genotype = newGenotype;
	}
}

void Evolver::epoch(std::mt19937 &generator) {
	normalizeFitness();

	std::vector<GenotypeAndFitness> newPopulation(_settings._populationSize);

	std::list<size_t> eliteIndices; // Used to keep from deleting elites later on

	// Add elites
	if (_settings._numElites != 0) {
		// Copy into list for best fitness search
		struct IndexAndFitness	{
			size_t _index;
			float _fitness;
		};

		std::list<size_t> populationIndices;

		for (size_t i = 0; i < _settings._populationSize; i++)
			populationIndices.push_back(i);

		// Find best and add directly into new population again
		for (size_t i = 0; i < _settings._numElites; i++) {
			// Find best
			std::list<size_t>::iterator it = populationIndices.begin();
			std::list<size_t>::iterator best = populationIndices.begin();

			it++;

			for (; it != populationIndices.end(); it++)
			if (_population[*it]._fitness > _population[*best]._fitness)
				best = it;

			eliteIndices.push_back(*best);
			newPopulation[i] = _population[*best];

			populationIndices.erase(best);
		}
	}

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	// Create rest of new population
	for (size_t i = _settings._numElites; i < _settings._populationSize; i++) {
		// Find parents
		size_t parentIndex1, parentIndex2;

		_selector->select(&_settings, _population, parentIndex1, parentIndex2, generator);

		// Create offspring
		std::shared_ptr<NetworkGenotype> child = _pGenotypeFactory();

		_population[parentIndex1]._genotype->crossover(*_population[parentIndex2]._genotype,
			*child, _settings._disableGeneChance,
			_population[parentIndex1]._fitness, _population[parentIndex2]._fitness, 
			_settings._minBias, _settings._maxBias, _functionChances,
			generator);

		child->crossoverAdditional(*_population[parentIndex2]._genotype,
			*child, _settings._disableGeneChance,
			_population[parentIndex1]._fitness, _population[parentIndex2]._fitness, generator);

		if (dist01(generator) < _settings._newNodeMutationRate)
			child->mutateAddNode(_settings._minWeight, _settings._maxWeight, _settings._minBias, _settings._maxBias, _functionChances, _innovationNumber, generator);

		if (dist01(generator) < _settings._newConnectionMutationRate)
			child->mutateAddConnection(_settings._minWeight, _settings._maxWeight, _settings._minBias, _settings._maxBias, _functionChances, _innovationNumber, generator);

		child->mutatePerturbWeight(_settings._weightPerturbationChance, _settings._maxPerturbation, generator);

		child->mutateChangeFunction(_settings._changeFunctionChance, _functionChances, generator);

		child->mutateAdditional(_innovationNumber, _settings._maxPerturbation, _settings._minWeight, _settings._maxWeight, _settings._minBias, _settings._maxBias, _innovationNumber, generator);

		newPopulation[i]._genotype = child;
	}

	// nullptrify references to elites
	for (std::list<size_t>::iterator it = eliteIndices.begin(); it != eliteIndices.end(); it++)
		_population[*it]._genotype = nullptr;

	// Destroy old population, with the exception of the elites since they are simply transfered
	clearPopulation();

	_population = newPopulation;
}