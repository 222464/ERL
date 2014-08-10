/*
ERL

Experiment
*/

#pragma once

#include <erl/field/Field2DCL.h>
#include <erl/platform/ComputeSystem.h>

namespace erl {
	class Experiment {
	protected:
		float _experimentWeight;

	public:
		Experiment()
			: _experimentWeight(1.0f)
		{}

		virtual float evaluate(Field2DGenes &fieldGenes, const Field2DEvolverSettings* pSettings,
			const std::shared_ptr<cl::Image2D> &randomImage,
			const std::shared_ptr<cl::Program> &blurProgram,
			const std::shared_ptr<cl::Kernel> &blurKernelX,
			const std::shared_ptr<cl::Kernel> &blurKernelY,
			const std::vector<std::function<float(float)>> &activationFunctions,
			const std::vector<std::string> &activationFunctionNames,
			float minInitRec, float maxInitRec, Logger &logger,
			ComputeSystem &cs, std::mt19937 &generator) = 0;

		float getExperimentWeight() const {
			return _experimentWeight;
		}
	};
}