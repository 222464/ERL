/*
ERL

Evolutionary trainer
*/

#pragma once

#include <erl/field/Field2DEvolver.h>
#include <erl/simulation/Experiment.h>

namespace erl {
	class EvolutionaryTrainer {
	private:
		std::vector<std::shared_ptr<Experiment>> _experiments;

		std::shared_ptr<cl::Image2D> _randomImage;
		std::shared_ptr<cl::Program> _blurProgram;
		std::shared_ptr<cl::Kernel> _blurKernelX;
		std::shared_ptr<cl::Kernel> _blurKernelY;

		std::vector<std::function<float(float)>> _activationFunctions;
		std::vector<std::string> _activationFunctionNames;
		float _minInitRec, _maxInitRec;

	public:
		Field2DEvolver _evolutionaryAlgorithm;

		size_t _runsPerExperiment;

		size_t _numElites;
		float _greedExponent;

		EvolutionaryTrainer();

		void create(size_t populationSize,
			const Field2DEvolverSettings* pSettings,
			const std::vector<float> &functionChances,
			const std::shared_ptr<cl::Image2D> &randomImage,
			const std::shared_ptr<cl::Program> &blurProgram,
			const std::shared_ptr<cl::Kernel> &blurKernelX,
			const std::shared_ptr<cl::Kernel> &blurKernelY,
			const std::vector<std::function<float(float)>> &activationFunctions,
			const std::vector<std::string> &activationFunctionNames,
			float minInitRec, float maxInitRec,
			std::mt19937 &generator);

		void evaluate(const Field2DEvolverSettings* pSettings,
			const std::vector<float> &functionChances, 
			ComputeSystem &cs, Logger &logger, std::mt19937 &generator);

		void normalizeFitnesses();

		void reproduce(const Field2DEvolverSettings* pSettings,
			const std::vector<float> &functionChances, std::mt19937 &generator);

		void writeBestToStream(std::ostream &os) const;

		float getBestFitness() const;
		float getAverageFitness() const;

		void addExperiment(const std::shared_ptr<Experiment> &experiment) {
			_experiments.push_back(experiment);
		}

		void removeExperiment(size_t index) {
			_experiments.erase(_experiments.begin() + index);
		}

		size_t getNumExperiments() const {
			return _experiments.size();
		}

		std::shared_ptr<Experiment> getExperiment(size_t index) {
			return _experiments[index];
		}
	};
}