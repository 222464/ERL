#include "Phenotype.h"

#include <assert.h>

using namespace ne;

Phenotype::Phenotype()
{}

void Phenotype::createFromGenotype(const Genotype &genotype) {
	_numInputs = genotype.getNumInputs();
	_numOutputs = genotype.getNumOutputs();

	std::unordered_map<size_t, size_t> inputNodeIDToInputIndex;

	for (size_t i = 0; i < genotype._inputNodeIDs.size(); i++)
		inputNodeIDToInputIndex[genotype._inputNodeIDs[i]] = i;

	// Flood fill backward from outputs to build minimal network
	std::list<size_t> openNodeIDs;

	std::unordered_map<size_t, size_t> nodeIDToIndex;

	std::unordered_set<size_t> recurrentNodeIndicesSet;

	for (size_t i = 0; i < genotype._outputNodeIDs.size(); i++)
		openNodeIDs.push_back(genotype._outputNodeIDs[i]);

	while (!openNodeIDs.empty()) {
		size_t currentNodeID = openNodeIDs.front();

		openNodeIDs.pop_front();

		std::shared_ptr<Node> newNode = std::make_shared<Node>();

		_nodes.push_back(newNode);

		nodeIDToIndex[currentNodeID] = _nodes.size() - 1;

		std::shared_ptr<const Genotype::Node> genotypeNode = genotype._nodes.at(currentNodeID);

		newNode->_bias = genotypeNode->_bias;
		newNode->_functionIndex = genotypeNode->_functionIndex;
		newNode->_output = 0.0f;

		for (std::unordered_map<size_t, float>::const_iterator cit0 = genotypeNode->_connections.begin(); cit0 != genotypeNode->_connections.end(); cit0++) {
			std::unordered_map<size_t, size_t>::const_iterator cit1 = nodeIDToIndex.find(cit0->first);

			Connection c;

			std::unordered_map<size_t, size_t>::const_iterator cit2 = inputNodeIDToInputIndex.find(cit0->first);

			if (cit2 == inputNodeIDToInputIndex.end()) {
				if (cit1 == nodeIDToIndex.end()) {
					c._fetchType = _intermediate;
					c._fetchIndex = cit0->first; // Set to ID for now, will be mapped to an index later (when all nodes have been explored)
				}
				else {
					c._fetchType = _recurrent;
					c._fetchIndex = cit1->second;
				}
			}
			else {
				c._fetchType = _input;

				c._fetchIndex = cit2->second;
			}
			
			c._weight = cit0->second;

			newNode->_connections.push_back(c);

			if (c._fetchType != _input) {
				if (cit1 == nodeIDToIndex.end()) {
					// Visit this node (if it exists)
					if (genotype._nodes.find(cit0->first) != genotype._nodes.end())
						openNodeIDs.push_back(cit0->first);
				}
				else { // Add as recurrent source node
					std::unordered_set<size_t>::const_iterator cit3 = recurrentNodeIndicesSet.find(cit1->second);

					if (cit3 == recurrentNodeIndicesSet.end()) {
						recurrentNodeIndicesSet.insert(cit1->second);
						_recurrentNodeIndices.push_back(cit1->second);
					}
				}
			}
		}
	}

	for (size_t i = 0; i < _nodes.size(); i++)
	for (size_t j = 0; j < _nodes[i]->_connections.size(); j++)
	if (_nodes[i]->_connections[j]._fetchType == _intermediate)
		_nodes[i]->_connections[j]._fetchIndex = nodeIDToIndex[_nodes[i]->_connections[j]._fetchIndex];

	// Flip nodes around
	std::reverse(_nodes.begin(), _nodes.end());

	// Convert node IDs in the phenotype to true node indices using conversion map. Take flipped node order into account. Do not change if it is an input connection
	for (size_t ni = 0; ni < _nodes.size(); ni++)
	for (size_t ci = 0; ci < _nodes[ni]->_connections.size(); ci++)
	if (_nodes[ni]->_connections[ci]._fetchType != _input) // If is input connection
		_nodes[ni]->_connections[ci]._fetchIndex = _nodes.size() - _nodes[ni]->_connections[ci]._fetchIndex - 1;

	for (size_t i = 0; i < _recurrentNodeIndices.size(); i++)
		_recurrentNodeIndices[i] = _nodes.size() - _recurrentNodeIndices[i] - 1;
}

void Phenotype::execute(const std::vector<float> &inputs, std::vector<float> &outputs, std::vector<float> &recurrentData, const std::vector<std::function<float(float)>> &functions) {
	assert(inputs.size() == _numInputs);
	assert(outputs.size() == _numOutputs);

	// Assign recurrent data
	for (size_t i = 0; i < _recurrentNodeIndices.size(); i++)
		_nodes[_recurrentNodeIndices[i]]->_output = recurrentData[i];

	for (size_t ni = 0; ni < _nodes.size(); ni++) {
		float sum = _nodes[ni]->_bias;

		for (size_t ci = 0; ci < _nodes[ni]->_connections.size(); ci++) {
			switch (_nodes[ni]->_connections[ci]._fetchType) {
			case _input:
				sum += _nodes[ni]->_connections[ci]._weight * inputs[_nodes[ni]->_connections[ci]._fetchIndex];

				break;
			case _recurrent:
			case _intermediate:
				sum += _nodes[ni]->_connections[ci]._weight * _nodes[_nodes[ni]->_connections[ci]._fetchIndex]->_output;

				break;
			}
		}

		_nodes[ni]->_output = functions[_nodes[ni]->_functionIndex](sum);
	}

	// Read recurrent data
	for (size_t i = 0; i < _recurrentNodeIndices.size(); i++)
		recurrentData[i] = _nodes[_recurrentNodeIndices[i]]->_output;

	// Read outputs
	const size_t numNonOutputNodes = _nodes.size() - _numOutputs;

	for (size_t i = 0; i < _numOutputs; i++)
		outputs[i] = _nodes[numNonOutputNodes + i]->_output;
}