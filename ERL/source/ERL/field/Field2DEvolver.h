#pragma once

#include <erl/field/Field2DGenes.h>

#include <functional>

namespace erl {
	class Field2DEvolver {
	private:
		void normalizeFitness(float greedExponent);
		size_t roulette(float totalFitness, std::mt19937 &generator);

		std::vector<std::shared_ptr<Field2DGenes>> _genotypes;
		std::vector<float> _fitnesses;

	public:
		Field2DEvolver();

		void create(size_t populationSize, const Field2DEvolverSettings* pSettings,
			const std::vector<float> &functionChances, std::mt19937 &generator);

		void setFitness(size_t index, float value) {
			_fitnesses[index] = value;
		}

		float getFitness(size_t index) const {
			return _fitnesses[index];
		}

		const std::shared_ptr<Field2DGenes> &getPopulationMember(size_t index) const {
			return _genotypes[index];
		}

		size_t getPopulationSize() const {
			return _genotypes.size();
		}

		void epoch(const Field2DEvolverSettings* pSettings, const std::vector<float> &functionChances, std::mt19937 &generator, size_t numElites, float greedExponent = 2.0f);
	};
}