/*
	NEAT Visualizer
	Copyright (C) 2012-2014 Eric Laukien

	This software is provided 'as-is', without any express or implied
	warranty.  In no event will the authors be held liable for any damages
	arising from the use of this software.

	Permission is granted to anyone to use this software for any purpose,
	including commercial applications, and to alter it and redistribute it
	freely, subject to the following restrictions:

	1. The origin of this software must not be misrepresented; you must not
		claim that you wrote the original software. If you use this software
		in a product, an acknowledgment in the product documentation would be
		appreciated but is not required.
	2. Altered source versions must be plainly marked as such, and must not be
		misrepresented as being the original software.
	3. This notice may not be removed or altered from any source distribution.

	This version of the NEAT Visualizer has been modified for ERL to include different activation functions (CPPN)
*/
#include <neat/NetworkPhenotype.h>

#include <list>

#include <assert.h>

#include <iostream>

using namespace neat;

NetworkPhenotype::NetworkPhenotype()
: _activationMultiplier(4.0f)
{}

NeuronInput &NetworkPhenotype::getNeuronInputNode(size_t index) {
	if (index >= _inputs.size()) {
		const size_t numHiddenAndInput = _inputs.size() + _hidden.size();

		if (index >= numHiddenAndInput)
			return _outputs[index - numHiddenAndInput];

		return _hidden[index - _inputs.size()];
	}

	return _inputs[index];
}

Neuron &NetworkPhenotype::getNeuronNode(size_t index) {
	assert(index >= _inputs.size());

	const size_t numHiddenAndInput = _inputs.size() + _hidden.size();

	if (index >= numHiddenAndInput)
		return _outputs[index - numHiddenAndInput];

	return _hidden[index - _inputs.size()];
}

void NetworkPhenotype::create(const NetworkGenotype &genotype) {
	size_t numInputs = genotype.getNumInputs();
	size_t numHidden = genotype.getNumHidden();
	size_t numOutputs = genotype.getNumOutputs();

	// Clear existing data, if there is any
	_inputs.clear();
	_hidden.clear();
	_outputs.clear();

	// Create neurons and neuron inputs
	_inputs.resize(numInputs);
	_hidden.resize(numHidden);
	_outputs.resize(numOutputs);

	const size_t numUnits = numInputs + numHidden + numOutputs;

	// Connect neurons
	for (size_t i = 0; i < genotype.getConnectionSet().size(); i++) {
		if (!genotype.getConnectionSet()[i]->_enabled) // Skip disabled genes
			continue;

		// Cannot have an output to a input
		if (genotype.getConnectionSet()[i]->_outIndex < _inputs.size() || genotype.getConnectionSet()[i]->_outIndex >= numUnits)
			continue;

		Neuron::Synapse newSynapse;
		newSynapse._inputOffset = genotype.getConnectionSet()[i]->_inIndex;
		newSynapse._weight = genotype.getConnectionSet()[i]->_weight;

		getNeuronNode(genotype.getConnectionSet()[i]->_outIndex)._inputs.push_back(newSynapse);
	}

	// Set node data
	for (size_t i = getNumInputs(); i < genotype.getNodeDataSize(); i++) {
		const NetworkGenotype::NodeData &data = genotype.getNodeData(i);

		Neuron &node = getNeuronNode(i);

		node._bias = data._bias;
		node._activationFunctionIndex = data._activationFunctionIndex;
	}
}

void NetworkPhenotype::update(const std::vector<std::function<float(float)>> &activationFunctions) {
	for (size_t i = 0, size = _hidden.size(); i < size; i++)
		_hidden[i].update(*this, activationFunctions);

	for (size_t i = 0, size = _outputs.size(); i < size; i++)
		_outputs[i].update(*this, activationFunctions);
}

void NetworkPhenotype::resetOutputs() {
	for (size_t i = 0, size = _hidden.size(); i < size; i++)
		_hidden[i]._output = 0.0f;

	for (size_t i = 0, size = _outputs.size(); i < size; i++)
		_outputs[i]._output = 0.0f;
}

void NetworkPhenotype::getConnectionData(std::unordered_set<Connection, Connection> &data, std::vector<std::vector<size_t>> &outgoingConnections, std::vector<bool> &recurrentSourceNodes) {
	size_t numNodes = getNumInputs() + getNumHidden() + getNumOutputs();

	outgoingConnections.resize(numNodes);

	recurrentSourceNodes.clear();
	recurrentSourceNodes.assign(numNodes, false);

	data.clear();

	for (size_t n = getNumInputs(); n < numNodes; n++) {
		const Neuron &neuron = getNeuronNode(n);

		for (size_t i = 0; i < neuron._inputs.size(); i++)
			outgoingConnections[neuron._inputs[i]._inputOffset].push_back(n);
	}

	std::vector<bool> explored(numNodes, false);

	std::list<size_t> queue;

	// Starting points of queue: all nodes without inputs (input nodes included)
	for (size_t n = 0; n < numNodes; n++)
	if (n < getNumInputs()) {
		queue.push_back(n);
		explored[n] = true;
	}
	else if (getNeuronNode(n)._inputs.empty()) {
		queue.push_back(n);
		//explored[n] = true;
	}

	while (!queue.empty()) {
		size_t current = queue.front();

		queue.pop_front();

		if (current >= getNumInputs()) {
			Neuron &neuron = getNeuronNode(current);

			for (size_t i = 0; i < neuron._inputs.size(); i++) {
				if (!explored[neuron._inputs[i]._inputOffset]) {
					recurrentSourceNodes[neuron._inputs[i]._inputOffset] = true;

					Connection c;

					c._inIndex = neuron._inputs[i]._inputOffset;
					c._outIndex = current;

					data.insert(c);
				}
			}
		}

		// Explore nodes whose inputs are this node
		for (size_t i = 0; i < outgoingConnections[current].size(); i++) {
			if (!explored[outgoingConnections[current][i]])
				queue.push_back(outgoingConnections[current][i]);
		}

		explored[current] = true;
	}
}