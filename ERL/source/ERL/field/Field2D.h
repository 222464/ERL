/*
ERL

Field2D
*/

#pragma once

#include <erl/platform/ComputeSystem.h>
#include <erl/field/Field2DGenes.h>

namespace erl {
	class Field2D {
	private:
		std::vector<cl::Image2D> _images;
	public:
		void create(const Field2DGenes &genes);
		void update(ComputeSystem &cs);
	};
}