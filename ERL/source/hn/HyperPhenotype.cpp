#include "HyperPhenotype.h"

#include <algorithm>

using namespace hn;

int hn::getIndex(const std::vector<int> &dimensions, const std::vector<int> &coordinate) {
	int index = 0;

	int size = 1;

	for (int d = 0; d < dimensions.size(); d++) {
		index += coordinate[d] * size;

		size *= dimensions[d];
	}

	return index;
}

void hn::getCoordinate(const std::vector<int> &dimensions, int index, std::vector<int> &coordinate) {
	assert(coordinate.size() == dimensions.size());

	int size = 1;

	for (int d = 0; d < dimensions.size(); d++) {
		coordinate[d] = (index / size) % dimensions[d];

		size *= dimensions[d];
	}
}

void HyperPhenotype::createFromGenotype(const ne::Genotype &connectionGenotype, const ne::Genotype &biasGenotype, const std::vector<int> &substrateDimensions, const std::vector<int> &inputIndices, const std::vector<std::function<float(float)>> &functions, int connectionRadius, float weightThreshold) {
	_substrateDimensions = substrateDimensions;
	_inputIndices = inputIndices;
	
	int totalSize = 1;

	for (int d = 0; d < _substrateDimensions.size(); d++)
		totalSize *= _substrateDimensions[d];

	_nodes.resize(totalSize);

	assert(connectionGenotype.getNumInputs() == 2 * _substrateDimensions.size());
	assert(connectionGenotype.getNumOutputs() == 1);

	assert(biasGenotype.getNumInputs() == _substrateDimensions.size());
	assert(biasGenotype.getNumOutputs() == 1);

	std::vector<float> substrateDimensionsInv(_substrateDimensions.size());

	for (int d = 0; d < substrateDimensionsInv.size(); d++)
		substrateDimensionsInv[d] = 1.0f / _substrateDimensions[d];

	float connectionRadiusInv = 1.0f / connectionRadius;

	ne::Phenotype connectionCPPN;
	connectionCPPN.createFromGenotype(connectionGenotype);

	ne::Phenotype biasCPPN;
	biasCPPN.createFromGenotype(biasGenotype);

	std::vector<float> connectionRecurrentData(connectionCPPN.getRecurrentDataSize(), 0.0f);
	std::vector<float> biasRecurrentData(biasCPPN.getRecurrentDataSize(), 0.0f);

	for (int i = 0; i < _nodes.size(); i++) {
		std::vector<int> nodeCoordinate(_substrateDimensions.size());

		getCoordinate(_substrateDimensions, i, nodeCoordinate);

		std::vector<float> biasInput(_substrateDimensions.size());

		for (int d = 0; d < _substrateDimensions.size(); d++)
			biasInput[d] = nodeCoordinate[d] * substrateDimensionsInv[d];

		std::vector<float> biasOutput(1);

		biasCPPN.execute(biasInput, biasOutput, biasRecurrentData, functions);

		_nodes[i]._bias = biasOutput[0];

		std::vector<int> lowerBound(_substrateDimensions.size());
		std::vector<int> upperBound(_substrateDimensions.size());

		for (int d = 0; d < _substrateDimensions.size(); d++) {
			lowerBound[d] = std::max(0, nodeCoordinate[d] - connectionRadius);
			upperBound[d] = std::min(_substrateDimensions[d] - 1, nodeCoordinate[d] + connectionRadius);
		}

		std::vector<int> connectionCoordinate = lowerBound;

		while (connectionCoordinate != upperBound) {
			std::vector<float> connectionInput(_substrateDimensions.size() * 2);

			for (int d = 0; d < _substrateDimensions.size(); d++)
				connectionInput[d] = biasInput[d];

			for (int d = 0; d < _substrateDimensions.size(); d++)
				connectionInput[_substrateDimensions.size() + d] = (connectionCoordinate[d] - nodeCoordinate[d]) * connectionRadiusInv;

			std::vector<float> connectionOutput(1);

			connectionCPPN.execute(connectionInput, connectionOutput, connectionRecurrentData, functions);

			if (std::abs(connectionOutput[0]) > weightThreshold) {
				Connection c;
				c._weight = connectionOutput[0];
				c._index = getIndex(_substrateDimensions, connectionCoordinate);

				_nodes[i]._connections.push_back(c);
			}

			// Increment coordinates
			connectionCoordinate[0]++;

			for (int d = 0; d < _substrateDimensions.size() - 1; d++)
				if (connectionCoordinate[d] > upperBound[d]) {
					connectionCoordinate[d] = lowerBound[d];
					connectionCoordinate[d + 1]++;
				}
				else
					break;
		}

		_nodes[i]._connections.shrink_to_fit();
	}

	for (int i = 0; i < _inputIndices.size(); i++)
		_nodes[_inputIndices[i]]._type = _input;
}

void HyperPhenotype::update() {
	for (int i = 0; i < _nodes.size(); i++) {
		if (_nodes[i]._type == _neuron) {
			float sum = _nodes[i]._bias;

			for (int j = 0; j < _nodes[i]._connections.size(); j++)
				sum += _nodes[i]._connections[j]._weight * _nodes[_nodes[i]._connections[j]._index]._statePrev;
			
			_nodes[i]._state = sigmoid(sum);
		}
	}

	for (int i = 0; i < _nodes.size(); i++)
		_nodes[i]._statePrev = _nodes[i]._state;
}

void HyperPhenotype::clearStates() {
	for (int i = 0; i < _nodes.size(); i++)
		_nodes[i]._statePrev = 0.0f;
}