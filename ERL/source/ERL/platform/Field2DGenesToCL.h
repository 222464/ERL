/*
ERL

Field2D genes to OpenCL
*/

#pragma once

#include <erl/field/Field2DGenes.h>
#include <string>

namespace erl {
	std::string field2DGenesNodeUpdateToCL(erl::Field2DGenes &genes, const std::vector<std::string> &functionNames, int fieldWidth, int fieldHeight, int connectionRadius);
}