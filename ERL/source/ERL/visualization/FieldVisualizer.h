/*
ERL

Field Visualizer
*/

#pragma once

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <erl/field/Field2D.h>

namespace erl {
	namespace fv {
		void show(sf::RenderTarget* pRenderTarget, const Field2D &field);
	}
}