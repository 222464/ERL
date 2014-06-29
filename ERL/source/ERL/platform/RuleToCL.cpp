#include <erl/platform/RuleToCL.h>

using namespace erl;

std::string getOutputNodeString(neat::NetworkPhenotype &phenotype, const std::unordered_set<neat::NetworkPhenotype::Connection, neat::NetworkPhenotype::Connection> &data,
	const std::vector<std::vector<size_t>> &outgoingConnections, const std::vector<bool> &recurrentSourceNodes, const std::vector<std::string> &functionNames,
	size_t nodeIndex);

void floodForwardFindCalculateIntermediates(neat::NetworkPhenotype &phenotype, const std::unordered_set<neat::NetworkPhenotype::Connection, neat::NetworkPhenotype::Connection> &data,
	const std::vector<std::vector<size_t>> &outgoingConnections, const std::vector<bool> &recurrentSourceNodes, const std::vector<std::string> &functionNames,
	size_t nodeIndex, std::vector<bool> &calculatedIntermediates, std::string &outputCode)
{
	// If is intermediate, compute it
	if (nodeIndex >= phenotype.getNumInputs()) {
		neat::Neuron &neuron = phenotype.getNeuronNode(nodeIndex);

		if (outgoingConnections[nodeIndex].size() >= 2 || neuron._inputs.empty()) {
			if (!calculatedIntermediates[nodeIndex]) {
				outputCode += "	float intermediate" + std::to_string(nodeIndex) + " = ";

				for (size_t i = 0; i < neuron._inputs.size(); i++) {
					neat::NetworkPhenotype::Connection c;
					c._inIndex = neuron._inputs[i]._inputOffset;
					c._outIndex = nodeIndex;

					std::unordered_set<neat::NetworkPhenotype::Connection, neat::NetworkPhenotype::Connection>::iterator it = data.find(c);

					if (it == data.end()) // Not recurrent connection
						outputCode += std::to_string(neuron._inputs[i]._weight) + " * " + getOutputNodeString(phenotype, data, outgoingConnections, recurrentSourceNodes, functionNames, neuron._inputs[i]._inputOffset);
					else
						outputCode += std::to_string(neuron._inputs[i]._weight) + " * (*recurrent" + std::to_string(neuron._inputs[i]._inputOffset) + ")";

					//if (i != neuron._inputs.size() - 1)
					outputCode += " + ";
				}

				outputCode += std::to_string(neuron._bias);

				outputCode += ";\n";

				calculatedIntermediates[nodeIndex] = true;
			}
		}
	}

	for (size_t i = 0; i < outgoingConnections[nodeIndex].size(); i++) {
		// Don't follow recurrent connections
		neat::NetworkPhenotype::Connection c;

		c._inIndex = nodeIndex;
		c._outIndex = outgoingConnections[nodeIndex][i];

		if (data.find(c) == data.end())
			floodForwardFindCalculateIntermediates(phenotype, data, outgoingConnections, recurrentSourceNodes, functionNames, outgoingConnections[nodeIndex][i], calculatedIntermediates, outputCode);
	}
}

std::string getOutputNodeString(neat::NetworkPhenotype &phenotype, const std::unordered_set<neat::NetworkPhenotype::Connection, neat::NetworkPhenotype::Connection> &data,
	const std::vector<std::vector<size_t>> &outgoingConnections, const std::vector<bool> &recurrentSourceNodes, const std::vector<std::string> &functionNames,
	size_t nodeIndex)
{
	if (nodeIndex < phenotype.getNumInputs())
		return "input" + std::to_string(nodeIndex);

	// If needs intermediate storage
	if (outgoingConnections[nodeIndex].size() >= 2)
		return "intermediate" + std::to_string(nodeIndex);

	const neat::Neuron &neuron = phenotype.getNeuronNode(nodeIndex);

	std::string sub = "";

	for (size_t i = 0; i < neuron._inputs.size(); i++) {
		neat::NetworkPhenotype::Connection c;
		c._inIndex = neuron._inputs[i]._inputOffset;
		c._outIndex = nodeIndex;

		std::unordered_set<neat::NetworkPhenotype::Connection, neat::NetworkPhenotype::Connection>::iterator it = data.find(c);

		if (it == data.end()) // Not recurrent connection
			sub += std::to_string(neuron._inputs[i]._weight) + " * " + getOutputNodeString(phenotype, data, outgoingConnections, recurrentSourceNodes, functionNames, neuron._inputs[i]._inputOffset);
		else
			sub += std::to_string(neuron._inputs[i]._weight) + " * (*recurrent" + std::to_string(neuron._inputs[i]._inputOffset) + ")";

		//if (i != neuron._inputs.size() - 1)
			sub += " + ";
	}

	sub += std::to_string(neuron._bias);

	return functionNames[neuron._activationFunctionIndex] + "(" + sub + ")";
}

std::string erl::ruleToCL(neat::NetworkPhenotype &phenotype, const std::string &ruleName, const std::string &bufferName, const std::vector<std::string> &functionNames) {
	size_t numNodes = phenotype.getNumInputs() + phenotype.getNumHidden() + phenotype.getNumOutputs();

	// Recurrent data
	std::unordered_set<neat::NetworkPhenotype::Connection, neat::NetworkPhenotype::Connection> data;
	std::vector<std::vector<size_t>> outgoingConnections;
	std::vector<bool> recurrentSourceNodes;

	phenotype.getConnectionData(data, outgoingConnections, recurrentSourceNodes);

	size_t numRecurrentNodes = 0;

	// Do not include inputs in recurrent node count
	for (size_t i = phenotype.getNumInputs(); i < recurrentSourceNodes.size(); i++)
	if (recurrentSourceNodes[i])
		numRecurrentNodes++;

	std::string code;

	code += "void " + ruleName + "(";

	// Inputs
	for (size_t i = 0; i < phenotype.getNumInputs(); i++) {
		code += "float input" + std::to_string(i) + ", ";
	}

	// Outputs
	size_t outputsStart = phenotype.getNumInputs() + phenotype.getNumHidden();

	for (size_t i = 0; i < phenotype.getNumOutputs(); i++) {
		code += "float* output" + std::to_string(i + outputsStart);
		
		if (numRecurrentNodes != 0 || i != phenotype.getNumOutputs() - 1)
			code += ", ";
	}

	// Recurrent
	for (size_t i = phenotype.getNumInputs(); i < recurrentSourceNodes.size(); i++) {
		if (recurrentSourceNodes[i]) {
			code += "float* recurrent" + std::to_string(i);

			code += ", ";
		}
	}

	// Erase last 2 characters, which are ", "
	code.pop_back();
	code.pop_back();

	code += ") {\n";

	std::vector<bool> calculatedIntermediates(numNodes, false);

	// Compute all intermediates
	for (size_t i = 0; i < phenotype.getNumInputs(); i++)
		floodForwardFindCalculateIntermediates(phenotype, data, outgoingConnections, recurrentSourceNodes, functionNames, i, calculatedIntermediates, code);

	for (size_t i = phenotype.getNumInputs(); i < numNodes; i++)
	if (phenotype.getNeuronNode(i)._inputs.empty())
		floodForwardFindCalculateIntermediates(phenotype, data, outgoingConnections, recurrentSourceNodes, functionNames, i, calculatedIntermediates, code);

	// Compute all outputs
	for (size_t i = 0; i < phenotype.getNumOutputs(); i++) {
		code += "	*output" + std::to_string(i + outputsStart) + " = " + getOutputNodeString(phenotype, data, outgoingConnections, recurrentSourceNodes, functionNames, i + outputsStart) + ";\n";
	}

	// Update recurrents
	for (size_t i = phenotype.getNumInputs(); i < recurrentSourceNodes.size(); i++) {
		if (recurrentSourceNodes[i]) {
			code += "	*recurrent" + std::to_string(i) + " = " + getOutputNodeString(phenotype, data, outgoingConnections, recurrentSourceNodes, functionNames, i) + ";\n";
		}
	}

	code += "}\n";

	return code;
}