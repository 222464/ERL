/*
ERL

Rule to OpenCL
*/

#pragma once

#include <neat/NetworkPhenotype.h>
#include <string>

namespace erl {
	std::string ruleToCL(neat::NetworkPhenotype &phenotype, const std::string &ruleName, const std::string &bufferName, const std::vector<std::string> &functionNames);
}