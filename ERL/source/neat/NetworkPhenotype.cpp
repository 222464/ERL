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
	// Clear existing data, if there is any
	_inputs.clear();
	_hidden.clear();
	_outputs.clear();

	// Create neurons and neuron inputs
	_inputs.resize(genotype.getNumInputs());
	_hidden.resize(genotype.getNumHidden());
	_outputs.resize(genotype.getNumOutputs());

	const size_t numUnits = genotype.getNumInputs() + genotype.getNumHidden() + genotype.getNumOutputs();
	const size_t numInputsAndHidden = genotype.getNumInputs() + genotype.getNumHidden();

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
	size_t totalNumNodes = getNumInputs() + getNumHidden() + getNumOutputs();

	for (size_t i = getNumInputs(); i < totalNumNodes; i++) {
		const NetworkGenotype::NodeData &data = genotype.getNodeData(i);

		Neuron &node = getNeuronNode(i);

		node._bias = data._bias;
		node._activationFunctionIndex = data._activationFunctionIndex;
	}

	// ------------------------------- Pruning -------------------------------

	// Flood fill to detect which connections and nodes contribute to output
	std::list<size_t> openList;

	for (size_t i = numInputsAndHidden; i < numUnits; i++)
		openList.push_back(i);

	std::vector<bool> marked(numUnits, false);

	for (size_t i = 0; i < getNumInputs(); i++)
		marked[i] = true;

	for (size_t i = 0; i < getNumOutputs(); i++)
		marked[numInputsAndHidden + i] = true;

	while (!openList.empty()) {
		size_t current = openList.front();

		openList.pop_front();

		// Add previous nodes connected to this one
		Neuron &node = getNeuronNode(current);

		for (size_t i = 0; i < node._inputs.size(); i++)
		if (!marked[node._inputs[i]._inputOffset]) {
			openList.push_back(node._inputs[i]._inputOffset);

			marked[node._inputs[i]._inputOffset] = true;
		}
	}

	// If nodes are not marked, remove connections to them
	for (size_t i = getNumInputs(); i < numUnits; i++) {
		Neuron &node = getNeuronNode(i);

		for (size_t j = 0; j < node._inputs.size();)
		if (!marked[node._inputs[j]._inputOffset])
			node._inputs.erase(node._inputs.begin() + j);
		else
			j++;
	}

	// Remove unmarked nodes
	std::vector<size_t> newIndices(numUnits);

	for (size_t i = 0; i < numUnits; i++)
		newIndices[i] = i;

	size_t assignOffset = 0;

	for (size_t i = getNumInputs(); i < numInputsAndHidden; i++) {
		size_t hiddenIndex = i - getNumInputs();
		
		while (i + assignOffset < numInputsAndHidden && !marked[i + assignOffset]) {
			assignOffset++;

			if (i + assignOffset < numInputsAndHidden)
				newIndices[i + assignOffset] = i;
		}

		if (i + assignOffset >= numInputsAndHidden)
			break;

		_hidden[hiddenIndex] = _hidden[hiddenIndex + assignOffset];
		marked[i] = marked[i + assignOffset];
	}


	// Shift output indices down
	for (size_t i = numInputsAndHidden; i < numUnits; i++)
		newIndices[i] -= assignOffset;

	for (size_t i = getNumInputs(); i < numUnits; i++) {
		Neuron &node = getNeuronNode(i);

		for (size_t j = 0; j < node._inputs.size(); j++) {
			node._inputs[j]._inputOffset = newIndices[node._inputs[j]._inputOffset];
		}
	}

	_hidden.resize(_hidden.size() - assignOffset);
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

void NetworkPhenotype::getConnectionData(RuleData &ruleData) {
	size_t numNodes = getNumInputs() + getNumHidden() + getNumOutputs();

	ruleData._outgoingConnections.resize(numNodes);

	ruleData._recurrentSourceNodes.clear();
	ruleData._recurrentSourceNodes.assign(numNodes, false);

	ruleData._data.clear();

	for (size_t n = getNumInputs(); n < numNodes; n++) {
		const Neuron &neuron = getNeuronNode(n);

		for (size_t i = 0; i < neuron._inputs.size(); i++)
			ruleData._outgoingConnections[neuron._inputs[i]._inputOffset].push_back(n);
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
					ruleData._recurrentSourceNodes[neuron._inputs[i]._inputOffset] = true;

					Connection c;

					c._inIndex = neuron._inputs[i]._inputOffset;
					c._outIndex = current;

					ruleData._data.insert(c);
				}
			}
		}

		// Explore nodes whose inputs are this node
		for (size_t i = 0; i < ruleData._outgoingConnections[current].size(); i++) {
			if (!explored[ruleData._outgoingConnections[current][i]])
				queue.push_back(ruleData._outgoingConnections[current][i]);
		}

		explored[current] = true;
	}

	// Do not include inputs in recurrent node count
	ruleData._numRecurrentSourceNodes = 0;

	for (size_t i = getNumInputs(); i < ruleData._recurrentSourceNodes.size(); i++)
	if (ruleData._recurrentSourceNodes[i])
		ruleData._numRecurrentSourceNodes++;
}