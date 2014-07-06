/*
ERL

Field Visualizer
*/

#pragma once

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <erl/platform/ComputeSystem.h>
#include <erl/field/Field2D.h>

namespace erl {
	class FieldVisualizer {
	private:
		// Visualization adapter (field to texture)
		cl::Program _adapterProgram;
		std::function<cl::Event(const cl::EnqueueArgs&, cl::Buffer&, cl::ImageGL&, int, int, int)> _adapterKernelFunctor;
		cl::ImageGL _adaptedImage;
		sf::Texture _adaptedTexture;

	public:
		bool create(ComputeSystem &cs, const std::string &adapterFileName, const Field2D &field, Logger &logger);

		void update(Field2D &field);

		sf::Texture &getTexture() {
			return _adaptedTexture;
		}
	};
}