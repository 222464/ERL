/*
ERL

Rule to OpenCL
*/

#pragma once

#include <ne/Phenotype.h>
#include <string>

namespace erl {
	std::string ruleToCL(ne::Phenotype &phenotype,
		const std::string &ruleName, const std::vector<std::string> &functionNames);
}