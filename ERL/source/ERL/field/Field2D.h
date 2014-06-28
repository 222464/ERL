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
		cl::Buffer _frontBuffer;
		cl::Buffer _backBuffer;

		cl::Program _program;

	public:
		void create(const Field2DGenes &genes);
		void update(ComputeSystem &cs);
	};
}