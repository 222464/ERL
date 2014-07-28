/*
ERL

Field2DCL genes to OpenCL
*/

#pragma once

#include <erl/field/Field2DGenes.h>
#include <erl/field/Field2DCL.h>
#include <neat/NetworkPhenotype.h>
#include <string>

namespace erl {
	std::string field2DGenesNodeUpdateToCL(erl::Field2DGenes &genes, const erl::Field2DCL &field,
		neat::NetworkPhenotype &connectionPhenotype, neat::NetworkPhenotype &nodePhenotype,
		const neat::NetworkPhenotype::RuleData &connectionRuleData, const neat::NetworkPhenotype::RuleData &nodeRuleData,
		const std::vector<std::string> &functionNames, int fieldWidth, int fieldHeight, int connectionRadius, int numInputs, int numOutputs);
}