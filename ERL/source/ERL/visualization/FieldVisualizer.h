/*
ERL

Field Visualizer
*/

#pragma once

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <erl/platform/ComputeSystem.h>
#include <erl/field/Field2DCL.h>

namespace erl {
	class FieldVisualizer {
	private:
		// Visualization adapter (field to texture)
		cl::Program _adapterProgram;
		cl::Kernel _adapterKernel;
		//std::function<cl::Event(const cl::EnqueueArgs&, cl::Buffer&, cl::Image2D&, int, int, int)> _adapterKernelFunctor;
		cl::Image2D _adaptedImage;
		SoftwareImage2D<sf::Color> _adaptedSoftImage;

	public:
		bool create(ComputeSystem &cs, const std::string &adapterFileName, const Field2DCL &field, Logger &logger);

		void update(ComputeSystem &cs, Field2DCL &field);

		const SoftwareImage2D<sf::Color> &getSoftImage() {
			return _adaptedSoftImage;
		}
	};
}