/*
ERL

Rule to OpenCL
*/

#pragma once

#include <neat/NetworkPhenotype.h>
#include <string>

namespace erl {
	std::string ruleToCL(neat::NetworkPhenotype &phenotype,
		std::unordered_set<neat::NetworkPhenotype::Connection, neat::NetworkPhenotype::Connection> &data,
		std::vector<std::vector<size_t>> &outgoingConnections,
		std::vector<bool> &recurrentSourceNodes,
		size_t &numRecurrentSourceNodes,
		const std::string &ruleName, const std::vector<std::string> &functionNames);
}