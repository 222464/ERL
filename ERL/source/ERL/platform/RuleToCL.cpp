#include <erl/platform/RuleToCL.h>

using namespace erl;

std::string getNodeString(neat::NetworkPhenotype &phenotype, size_t nodeIndex) {


}

std::string erl::ruleToCL(neat::NetworkPhenotype &phenotype, const std::string &ruleName, const std::string &bufferName, std::vector<std::string> &functionNames) {
	std::string code;

	code += "void " + ruleName + "(";

	// Inputs
	for (size_t i = 0; i < phenotype.getNumInputs(); i++) {
		code += "float i" + std::to_string(i) + ", ";
	}

	// Mark values that need to be stored as intermediates


	// Recurrent values
	std::vector<neat::NetworkPhenotype::ConnectionData> data;

	for (size_t i = 0; i < phenotype.getNumOutputs(); i++)

}