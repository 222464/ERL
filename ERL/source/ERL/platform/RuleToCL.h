/*
ERL

Rule to OpenCL
*/

#pragma once

#include <neat/NetworkPhenotype.h>
#include <string>

namespace erl {
	std::string ruleToCL(neat::NetworkPhenotype &phenotype,
		const neat::NetworkPhenotype::RuleData &ruleData,
		const std::string &ruleName, const std::vector<std::string> &functionNames);
}