#include <erl/platform/RuleToCL.h>

using namespace erl;

struct ConnectionDesc {
	size_t _inIndex;
	size_t _outIndex;

	//ne::Phenotype::FetchType _fetchType;

	bool operator==(const ConnectionDesc &other) const {
		return _inIndex == other._inIndex && _outIndex == other._outIndex;// && _fetchType == other._fetchType;
	}

	size_t operator()(const ConnectionDesc &c) const {
		return c._inIndex ^ c._outIndex;// ^ _fetchType;
	}
};

std::string getOutputNodeString(ne::Phenotype &phenotype, const std::vector<std::vector<size_t>> &outgoingConnectionsInput, const std::vector<std::vector<size_t>> &outgoingConnectionsRecurrentIntermediate, std::unordered_set<ConnectionDesc, ConnectionDesc> &data,
	const std::vector<std::string> &functionNames, std::vector<bool> &calculatedIntermediates, int nodeIndex);

void floodForwardCalculateIntermediates(std::list<int> &openList, ne::Phenotype &phenotype, const std::vector<std::vector<size_t>> &outgoingConnectionsInput, const std::vector<std::vector<size_t>> &outgoingConnectionsRecurrentIntermediate, std::unordered_set<ConnectionDesc, ConnectionDesc> &data,
	const std::vector<std::string> &functionNames, std::vector<bool> &calculatedIntermediates, std::string &outputCode)
{
	int nodeIndex = openList.front();

	openList.pop_front();

	// If is intermediate, compute it
	if (nodeIndex >= 0) {
		std::shared_ptr<ne::Phenotype::Node> neuron = phenotype.getNodes()[nodeIndex];

		size_t numNonRecurrentOutgoingConnections = 0;

		for (size_t i = 0; i < outgoingConnectionsRecurrentIntermediate[nodeIndex].size(); i++) {
			ConnectionDesc cd;

			cd._inIndex = nodeIndex;
			cd._outIndex = outgoingConnectionsRecurrentIntermediate[nodeIndex][i];

			if (data.find(cd) == data.end())
				numNonRecurrentOutgoingConnections++;
		}

		if (numNonRecurrentOutgoingConnections >= 2) { //  || neuron._inputs.empty()
			if (!calculatedIntermediates[nodeIndex]) {
				outputCode += "	float intermediate" + std::to_string(nodeIndex) + " = ";

				for (size_t i = 0; i < neuron->_connections.size(); i++) {
					const ne::Phenotype::Connection &c = neuron->_connections[i];
					ConnectionDesc cd;

					cd._inIndex = c._fetchIndex;
					cd._outIndex = nodeIndex;

					std::unordered_set<ConnectionDesc, ConnectionDesc>::iterator it = data.find(cd);

					if (it == data.end()) // Not recurrent connection
						outputCode += std::to_string(neuron->_connections[i]._weight) + "f * " + getOutputNodeString(phenotype, outgoingConnectionsInput, outgoingConnectionsRecurrentIntermediate, data, functionNames, calculatedIntermediates, neuron->_connections[i]._fetchType == ne::Phenotype::_input ? -static_cast<int>(neuron->_connections[i]._fetchIndex) - 1 : neuron->_connections[i]._fetchIndex);
					else
						outputCode += std::to_string(neuron->_connections[i]._weight) + "f * (*recurrent" + std::to_string(neuron->_connections[i]._fetchIndex) + ")";

					//if (i != neuron._inputs.size() - 1)
					outputCode += " + ";
				}

				outputCode += std::to_string(neuron->_bias) + "f";

				outputCode += ";\n";

				calculatedIntermediates[nodeIndex] = true;
			}
		}

		for (size_t i = 0; i < outgoingConnectionsRecurrentIntermediate[nodeIndex].size(); i++) {
			// Don't follow recurrent connections
			ConnectionDesc cd;

			cd._inIndex = nodeIndex;
			cd._outIndex = outgoingConnectionsRecurrentIntermediate[nodeIndex][i];

			if (data.find(cd) == data.end())
				openList.push_back(outgoingConnectionsRecurrentIntermediate[nodeIndex][i]);
		}
	}
	else {
		size_t inputIndex = -nodeIndex - 1;

		for (size_t i = 0; i < outgoingConnectionsInput[inputIndex].size(); i++)
			openList.push_back(outgoingConnectionsInput[inputIndex][i]);
	}
}

std::string getOutputNodeString(ne::Phenotype &phenotype, const std::vector<std::vector<size_t>> &outgoingConnectionsInput, const std::vector<std::vector<size_t>> &outgoingConnectionsRecurrentIntermediate, std::unordered_set<ConnectionDesc, ConnectionDesc> &data,
	const std::vector<std::string> &functionNames, std::vector<bool> &calculatedIntermediates, int nodeIndex)
{
	if (nodeIndex < 0)
		return "input" + std::to_string(-nodeIndex - 1);

	size_t numNonRecurrentOutgoingConnections = 0;

	for (size_t i = 0; i < outgoingConnectionsRecurrentIntermediate[nodeIndex].size(); i++) {
		ConnectionDesc cd;

		cd._inIndex = nodeIndex;
		cd._outIndex = outgoingConnectionsRecurrentIntermediate[nodeIndex][i];

		if (data.find(cd) == data.end())
			numNonRecurrentOutgoingConnections++;
	}

	// If needs intermediate storage
	if (numNonRecurrentOutgoingConnections >= 2 && calculatedIntermediates[nodeIndex])
		return "intermediate" + std::to_string(nodeIndex);

	std::shared_ptr<ne::Phenotype::Node> neuron = phenotype.getNodes()[nodeIndex];

	std::string sub = "";

	for (size_t i = 0; i < neuron->_connections.size(); i++) {
		ConnectionDesc cd;
		cd._inIndex = neuron->_connections[i]._fetchIndex;
		cd._outIndex = nodeIndex;

		std::unordered_set<ConnectionDesc, ConnectionDesc>::iterator it = data.find(cd);

		if (it == data.end()) // Not recurrent connection
			sub += std::to_string(neuron->_connections[i]._weight) + "f * " + getOutputNodeString(phenotype, outgoingConnectionsInput, outgoingConnectionsRecurrentIntermediate, data, functionNames, calculatedIntermediates, neuron->_connections[i]._fetchType == ne::Phenotype::_input ? -static_cast<int>(neuron->_connections[i]._fetchIndex) - 1 : neuron->_connections[i]._fetchIndex);
		else
			sub += std::to_string(neuron->_connections[i]._weight) + "f * (*recurrent" + std::to_string(neuron->_connections[i]._fetchIndex) + ")";

		//if (i != neuron._inputs.size() - 1)
			sub += " + ";
	}

	sub += std::to_string(neuron->_bias) + "f";

	return functionNames[neuron->_functionIndex] + "(" + sub + ")";
}

std::string erl::ruleToCL(ne::Phenotype &phenotype,
	const std::string &ruleName, const std::vector<std::string> &functionNames)
{
	size_t numNodes = phenotype.getNodes().size();
	size_t numHidden = numNodes - phenotype.getNumOutputs();

	size_t numRecurrentNodes = 0;

	// Calculate connection data set and outgoing connections
	std::unordered_set<ConnectionDesc, ConnectionDesc> data;
	std::vector<std::vector<size_t>> outgoingConnectionsInput;
	std::vector<std::vector<size_t>> outgoingConnectionsRecurrentIntermediate;

	outgoingConnectionsInput.resize(phenotype.getNumInputs());
	outgoingConnectionsRecurrentIntermediate.resize(numNodes);

	for (size_t i = 0; i < numNodes; i++) {
		std::shared_ptr<ne::Phenotype::Node> neuron = phenotype.getNodes()[i];

		for (size_t j = 0; j < neuron->_connections.size(); j++) {
			ConnectionDesc cd;

			cd._inIndex = neuron->_connections[j]._fetchIndex;
			cd._outIndex = i;

			if (neuron->_connections[j]._fetchType == ne::Phenotype::_input)
				outgoingConnectionsInput[cd._inIndex].push_back(i);
			else
				outgoingConnectionsRecurrentIntermediate[cd._inIndex].push_back(i);
		}
	}

	std::unordered_set<size_t> recurrentSourceNodesSet;

	for (size_t i = 0; i < phenotype.getRecurrentNodeIndices().size(); i++)
		recurrentSourceNodesSet.insert(phenotype.getRecurrentNodeIndices()[i]);

	for (size_t n = 0; n < numNodes; n++) {
		for (size_t c = 0; c < phenotype.getNodes()[n]->_connections.size(); c++) {
			const ne::Phenotype::Connection &connection = phenotype.getNodes()[n]->_connections[c];

			if (connection._fetchType != ne::Phenotype::_input) {
				if (recurrentSourceNodesSet.find(connection._fetchIndex) != recurrentSourceNodesSet.end()) {
					ConnectionDesc cd;

					cd._inIndex = connection._fetchIndex;
					cd._outIndex = n;

					data.insert(cd);
				}
			}
		}
	}

	// Do not include inputs in recurrent node count
	std::string code;

	code += "void " + ruleName + "(";

	// Inputs
	for (size_t i = 0; i < phenotype.getNumInputs(); i++) {
		code += "float input" + std::to_string(i) + ", ";
	}

	// Outputs
	size_t outputsStart = numHidden;

	for (size_t i = 0; i < phenotype.getNumOutputs(); i++) {
		code += "float* output" + std::to_string(i) + ", ";
		
		//if (numRecurrentNodes != 0 || i != phenotype.getNumOutputs() - 1)
		//	code += ", ";
	}

	// Recurrent
	for (size_t i = 0; i < phenotype.getRecurrentNodeIndices().size(); i++) {
		code += "float* recurrent" + std::to_string(phenotype.getRecurrentNodeIndices()[i]);

		code += ", ";
	}

	// Erase last 2 characters, which are ", "
	code.pop_back();
	code.pop_back();

	code += ") {\n";

	std::vector<bool> calculatedIntermediates(numNodes, false);

	// Compute all intermediates
	for (size_t i = 0; i < phenotype.getNumInputs(); i++) {
		int startIndex = -static_cast<int>(i)-1;

		std::list<int> openList;

		openList.push_back(startIndex);

		while (!openList.empty())
			floodForwardCalculateIntermediates(openList, phenotype, outgoingConnectionsInput, outgoingConnectionsRecurrentIntermediate, data, functionNames, calculatedIntermediates, code);
	}

	for (size_t i = 0; i < numNodes; i++)
	if (phenotype.getNodes()[i]->_connections.empty()) {
		int startIndex = i;

		std::list<int> openList;

		openList.push_back(startIndex);

		while (!openList.empty())
			floodForwardCalculateIntermediates(openList, phenotype, outgoingConnectionsInput, outgoingConnectionsRecurrentIntermediate, data, functionNames, calculatedIntermediates, code);
	}

	// Compute all outputs
	for (size_t i = 0; i < phenotype.getNumOutputs(); i++) {
		code += "	(*output" + std::to_string(i) + ") = " + getOutputNodeString(phenotype, outgoingConnectionsInput, outgoingConnectionsRecurrentIntermediate, data, functionNames, calculatedIntermediates, i) + ";\n";
	}

	// Update recurrents
	for (size_t i = 0; i < phenotype.getRecurrentNodeIndices().size(); i++) {
		code += "	(*recurrent" + std::to_string(phenotype.getRecurrentNodeIndices()[i]) + ") = " + getOutputNodeString(phenotype, outgoingConnectionsInput, outgoingConnectionsRecurrentIntermediate, data, functionNames, calculatedIntermediates, phenotype.getRecurrentNodeIndices()[i]) + ";\n";
	}

	code += "}\n";

	return code;
}