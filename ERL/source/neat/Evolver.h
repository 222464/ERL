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

#include <neat/NetworkGenotype.h>

#include <neat/Evolvable.h>

#include <functional>

namespace neat {
	class EvolverSettings {
	public:
		float _speciationTolerance;
		float _preferSimilarFactor;
		float _reproduceRatio;

		float _newConnectionMutationRate;
		float _newNodeMutationRate;
		float _weightPerturbationChance;
		float _disableGeneChance;

		float _minInitWeight;
		float _maxInitWeight;
		float _minInitBias;
		float _maxInitBias;
		float _maxPerturbation;
		float _changeFunctionChance;

		float _minWeight;
		float _maxWeight;
		float _minBias;
		float _maxBias;

		// Species similarity weighting factors
		float _excessFactor;
		float _disjointFactor;
		float _averageWeightDifferenceFactor;
		float _inputCountDifferenceFactor;
		float _outputCountDifferenceFactor;
		float _functionFactor;

		size_t _populationSize;

		size_t _numElites;

		EvolverSettings();
		virtual ~EvolverSettings() {}
	};

	class Evolver {
	private:
		size_t _numInputs, _numOutputs;

		std::vector<float> _functionChances;

		std::function<std::shared_ptr<Evolvable>()> _genotypeFactory;

		std::shared_ptr<EvolverSettings> _settings;

		void normalizeFitness();

	public:
		struct GenotypeAndFitness {
			std::shared_ptr<Evolvable> _genotype;
			float _fitness;
		};

		std::shared_ptr<class ParentSelector> _selector;

		InnovationNumberType _innovationNumber;

		std::vector<GenotypeAndFitness> _population;

		Evolver();

		void initialize(size_t numInputs, size_t numOutputs,
			const std::vector<float> &functionChances, std::shared_ptr<class ParentSelector> selector, std::mt19937 &generator,
			const std::shared_ptr<EvolverSettings> &settings, std::function<std::shared_ptr<Evolvable>()> genotypeFactory = defaultGenotypeFactory);

		void clearPopulation();

		size_t getPopulationSize() const {
			return _population.size();
		}

		void epoch(std::mt19937 &generator); // Assumes proper fitness levels for this epoch have been set already

		size_t getNumInputs() const {
			return _numInputs;
		}

		size_t getNumOutputs() const {
			return _numOutputs;
		}

		static std::shared_ptr<Evolvable> defaultGenotypeFactory() {
			return std::shared_ptr<Evolvable>(new NetworkGenotype());
		}
	};
}