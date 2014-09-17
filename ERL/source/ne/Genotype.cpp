#include "Genotype.h"

#include <numeric>
#include <algorithm>
#include <assert.h>

using namespace ne;

size_t ne::roulette(const std::vector<float> &chances, std::mt19937 &generator) {
	float sum = std::accumulate(chances.begin(), chances.end(), 0.0f);

	std::uniform_real_distribution<float> cuspDist(0.0f, sum);

	float cusp = cuspDist(generator);

	float sumSoFar = 0.0f;

	for (size_t i = 0; i < chances.size(); i++) {
		sumSoFar += chances[i];

		if (sumSoFar >= cusp)
			return i;
	}

	return 0;
}

const Genotype &Genotype::operator=(const Genotype &other) {
	_nodes.clear();

	_inputNodeIDs = other._inputNodeIDs;
	_outputNodeIDs = other._outputNodeIDs;

	for (std::unordered_map<size_t, std::shared_ptr<Node>>::const_iterator cit0 = other._nodes.begin(); cit0 != other._nodes.end(); cit0++)
		_nodes[cit0->first] = std::shared_ptr<Node>(new Node(*cit0->second));

	_nextNodeID = other._nextNodeID;

	return *this;
}

float Genotype::getDifference(const Genotype &genotype0, const Genotype &genotype1, size_t node0ID, size_t node1ID, int searchDepth, float importanceDecay, float weightFactor, float disjointFactor, const std::unordered_map<FunctionPair, float, FunctionPair> &functionFactors, std::unordered_set<size_t> &visitedNodeIDs) {
	int disjointConnections0 = 0;
	int disjointConnections1 = 0;

	float weightDifference = 0.0f;

	float connectedNodeDifference = 0.0f;

	std::shared_ptr<Node> node0 = genotype0._nodes.at(node0ID);
	std::shared_ptr<Node> node1 = genotype1._nodes.at(node1ID);

	for (std::unordered_map<size_t, float>::const_iterator cit0 = node0->_connections.begin(); cit0 != node0->_connections.end(); cit0++) {
		std::unordered_map<size_t, float>::const_iterator cit1 = node1->_connections.find(cit0->first);

		if (cit1 == node1->_connections.end())
			disjointConnections0++;
		else {
			weightDifference += std::abs(cit0->second - cit1->second);

			visitedNodeIDs.insert(node0ID);
			visitedNodeIDs.insert(node1ID);

			if (searchDepth != 0)
				connectedNodeDifference += getDifference(genotype0, genotype1, cit0->first, cit1->first, searchDepth - 1, importanceDecay, weightFactor, disjointFactor, functionFactors, visitedNodeIDs);
		}
	}

	for (std::unordered_map<size_t, float>::const_iterator cit1 = node1->_connections.begin(); cit1 != node1->_connections.end(); cit1++) {
		std::unordered_map<size_t, float>::const_iterator cit0 = node1->_connections.find(cit1->first);

		if (cit0 == node0->_connections.end())
			disjointConnections1++;
	}

	FunctionPair fPair;

	fPair._functionIndex0 = node0->_functionIndex;
	fPair._functionIndex1 = node1->_functionIndex;

	float functionDifference = functionFactors.at(fPair);

	return weightFactor * weightDifference + disjointFactor * (disjointConnections0 + disjointConnections1) + importanceDecay * connectedNodeDifference + functionDifference;
}

void Genotype::createRandomFeedForward(size_t numInputs, size_t numOutputs, float minWeight, float maxWeight, const std::vector<float> &functionChances, std::mt19937 &generator) {
	_nodes.clear();
	
	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	size_t numNodes = numInputs + numOutputs;

	for (size_t ni = 0; ni < numNodes; ni++) {
		std::shared_ptr<Node> node = _nodes[ni] = std::make_shared<Node>();

		node->_bias = weightDist(generator);
		node->_functionIndex = roulette(functionChances, generator);
	}

	for (size_t ni = numInputs; ni < numNodes; ni++)
	for (size_t ci = 0; ci < numInputs; ci++)
		_nodes[ni]->_connections[ci] = weightDist(generator);

	_inputNodeIDs.resize(numInputs);

	for (size_t ni = 0; ni < numInputs; ni++) {
		for (size_t ci = 0; ci < numOutputs; ci++)
			_nodes[ni]->_outputNodes.insert(numInputs + ci);

		_inputNodeIDs[ni] = ni;
	}

	_outputNodeIDs.resize(numOutputs);

	for (size_t i = 0; i < numOutputs; i++)
		_outputNodeIDs[i] = numInputs + i;

	_nextNodeID = numNodes;
}

void Genotype::addNode(float minWeight, float maxWeight, const std::vector<float> &functionChances, std::mt19937 &generator) {
	// Select random node
	size_t numNodesWithConnections = 0;

	for (std::unordered_map<size_t, std::shared_ptr<Node>>::const_iterator cit0 = _nodes.begin(); cit0 != _nodes.end(); cit0++)
	if (!cit0->second->_connections.empty())
		numNodesWithConnections++;

	assert(numNodesWithConnections > 0);

	std::uniform_int_distribution<int> nodeIndexDist(0, numNodesWithConnections - 1);

	size_t nodeIndex = nodeIndexDist(generator);

	std::unordered_map<size_t, std::shared_ptr<Node>>::iterator it0 = _nodes.begin();

	size_t ni = 0;

	while (true) {
		while (it0->second->_connections.empty())
			it0++;

		if (ni < nodeIndex) {
			it0++;

			ni++;
		}
		else
			break;
	}

	// Select random connection to split
	std::uniform_int_distribution<int> connectionIndexDist(0, it0->second->_connections.size() - 1);

	size_t connectionIndex = connectionIndexDist(generator);

	std::unordered_map<size_t, float>::iterator it1 = it0->second->_connections.begin();

	for (size_t i = 0; i < connectionIndex; i++)
		it1++;

	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	size_t newNodeID = _nextNodeID;
	size_t oldConnectionNodeID = it1->first;

	std::shared_ptr<Node> newNode = std::make_shared<Node>();

	newNode->_bias = weightDist(generator);
	newNode->_functionIndex = roulette(functionChances, generator);
	newNode->_connections[oldConnectionNodeID] = weightDist(generator);

	float oldConnectionWeight = it1->second;

	it0->second->_connections.erase(it1);

	it0->second->_connections[newNodeID] = oldConnectionWeight;

	_nodes[newNodeID] = newNode;

	newNode->_outputNodes.insert(it0->first);

	std::shared_ptr<Node> oldNode = _nodes[oldConnectionNodeID];

	oldNode->_outputNodes.erase(oldNode->_outputNodes.find(it0->first));
	oldNode->_outputNodes.insert(newNodeID);

	_nextNodeID++;
}

void Genotype::addConnection(float minWeight, float maxWeight, std::mt19937 &generator) {
	assert(!_nodes.empty());

	size_t numFullyConnected = 0;

	// Construct set of inputs
	std::unordered_set<size_t> inputNodeIDSet;

	for (size_t i = 0; i < getNumInputs(); i++)
		inputNodeIDSet.insert(_inputNodeIDs[i]);

	for (std::unordered_map<size_t, std::shared_ptr<Node>>::const_iterator cit0 = _nodes.begin(); cit0 != _nodes.end(); cit0++)
	if (inputNodeIDSet.find(cit0->first) != inputNodeIDSet.end() || cit0->second->_connections.size() >= _nodes.size())
		numFullyConnected++;

	if (numFullyConnected == _nodes.size())
		return;

	// Select random node that is not an input and isn't fully connected
	std::uniform_int_distribution<int> nodeIndexDist(0, _nodes.size() - numFullyConnected - 1);

	size_t nodeIndex = nodeIndexDist(generator);

	std::unordered_map<size_t, std::shared_ptr<Node>>::iterator it0 = _nodes.begin();

	size_t ni = 0;

	while (true) {
		while (inputNodeIDSet.find(it0->first) != inputNodeIDSet.end() || it0->second->_connections.size() >= _nodes.size())
			it0++;

		if (ni < nodeIndex) {
			it0++;

			ni++;
		}
		else
			break;
	}

	std::vector<size_t> connectionNodeIDs;

	for (std::unordered_map<size_t, std::shared_ptr<Node>>::iterator it1 = _nodes.begin(); it1 != _nodes.end(); it1++)
	if (it0->second->_connections.find(it1->first) == it0->second->_connections.end())
		connectionNodeIDs.push_back(it1->first);

	std::uniform_int_distribution<int> connectionIndexDist(0, connectionNodeIDs.size() - 1);
	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	size_t connectionIndex = connectionIndexDist(generator);

	it0->second->_connections[connectionNodeIDs[connectionIndex]] = weightDist(generator);
	_nodes[connectionNodeIDs[connectionIndex]]->_outputNodes.insert(it0->first);
}

void Genotype::createFromParents(const Genotype &parent0, const Genotype &parent1, float averageChance, std::mt19937 &generator) {
	_nodes.clear();

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	
	// For each node, find the node that is most similar, then remove these nodes so they cannot be used again. Repeat until either parent is out of nodes.
	std::list<size_t> parent0NodeIDs;
	std::unordered_set<size_t> parent1NodeIDs;

	for (std::unordered_map<size_t, std::shared_ptr<Node>>::const_iterator cit0 = parent0._nodes.begin(); cit0 != parent0._nodes.end(); cit0++)
		parent0NodeIDs.push_back(cit0->first);

	for (std::unordered_map<size_t, std::shared_ptr<Node>>::const_iterator cit1 = parent1._nodes.begin(); cit1 != parent1._nodes.end(); cit1++)
		parent1NodeIDs.insert(cit1->first);

	for (std::list<size_t>::iterator nIDIt0 = parent0NodeIDs.begin(); nIDIt0 != parent0NodeIDs.end();) {
		std::unordered_set<size_t>::iterator nIDIt1 = parent1NodeIDs.find(*nIDIt0);

		if (nIDIt1 != parent1NodeIDs.end()) {
			std::shared_ptr<Node> node0 = parent0._nodes.at(*nIDIt0);
			std::shared_ptr<Node> node1 = parent1._nodes.at(*nIDIt1);

			// Merge these nodes
			assert(*nIDIt0 == *nIDIt1);

			size_t ID = *nIDIt0;

			std::shared_ptr<Node> child = std::make_shared<Node>();

			child->_bias = dist01(generator) < averageChance ? (node0->_bias + node1->_bias) * 0.5f : (dist01(generator) < 0.5f ? node0->_bias : node1->_bias);
			child->_functionIndex = dist01(generator) < 0.5f ? node0->_functionIndex : node1->_functionIndex;

			for (std::unordered_map<size_t, float>::const_iterator cit0 = node0->_connections.begin(); cit0 != node0->_connections.end(); cit0++) {
				std::unordered_map<size_t, float>::const_iterator cit1 = node1->_connections.find(cit0->first);

				if (cit1 != node1->_connections.end()) {
					// Add connection
					float weight = dist01(generator) < averageChance ? (cit0->second + cit1->second) * 0.5f : (dist01(generator) < 0.5f ? cit0->second : cit1->second);

					child->_connections[cit0->first] = weight;
				}
				else
					// This is a disjoint connection, add
					child->_connections[cit0->first] = cit0->second;
			}

			for (std::unordered_map<size_t, float>::const_iterator cit1 = node1->_connections.begin(); cit1 != node1->_connections.end(); cit1++) {
				std::unordered_map<size_t, float>::const_iterator cit0 = node0->_connections.find(cit1->first);

				// This is a disjoint connection, add
				if (cit0 == node0->_connections.end())
					child->_connections[cit1->first] = cit1->second;
			}

			_nodes[ID] = child;

			// Remove these nodes from their respective lists
			nIDIt0 = parent0NodeIDs.erase(nIDIt0);
			parent1NodeIDs.erase(nIDIt1);
		}
		else
			nIDIt0++;
	}

	// Check if there are leftover nodes. If there are, add them all
	for (std::list<size_t>::iterator it = parent0NodeIDs.begin(); it != parent0NodeIDs.end(); it++)
		_nodes[*it].reset(new Node(*parent0._nodes.at(*it)));

	for (std::unordered_set<size_t>::iterator it = parent1NodeIDs.begin(); it != parent1NodeIDs.end(); it++)
		_nodes[*it].reset(new Node(*parent1._nodes.at(*it)));

	_nextNodeID = std::max(parent0._nextNodeID, parent1._nextNodeID);

	// Calculate outgoing connections
	for (std::unordered_map<size_t, std::shared_ptr<Node>>::iterator it0 = _nodes.begin(); it0 != _nodes.end(); it0++)
	for (std::unordered_map<size_t, float>::iterator it1 = it0->second->_connections.begin(); it1 != it0->second->_connections.end();) {
		std::unordered_map<size_t, std::shared_ptr<Node>>::iterator it2 = _nodes.find(it1->first);

		if (it2 == _nodes.end())
			it1 = it0->second->_connections.erase(it1);
		else {
			_nodes[it1->first]->_outputNodes.insert(it0->first);

			it1++;
		}
	}

	calculateOutgoingConnections();
}

void Genotype::mutate(float addNodeChance, float addConnectionChance, float minWeight, float maxWeight, float perturbationChance, float perturbationStdDev, float changeFunctionChance, const std::vector<float> &functionChances, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> distPerturbation(0.0f, perturbationStdDev);
	
	// Mutate existing connections
	for (std::unordered_map<size_t, std::shared_ptr<Node>>::iterator it0 = _nodes.begin(); it0 != _nodes.end(); it0++) {
		for (std::unordered_map<size_t, float>::iterator it1 = it0->second->_connections.begin(); it1 != it0->second->_connections.end(); it1++)
		if (dist01(generator) < perturbationChance)
			it1->second += distPerturbation(generator);

		if (dist01(generator) < perturbationChance)
			it0->second->_bias += distPerturbation(generator);

		if (dist01(generator) < changeFunctionChance)
			it0->second->_functionIndex = roulette(functionChances, generator);
	}

	if (dist01(generator) < addNodeChance)
		addNode(minWeight, maxWeight, functionChances, generator);

	if (dist01(generator) < addConnectionChance)
		addConnection(minWeight, maxWeight, generator);

	calculateOutgoingConnections();
}

float Genotype::getDifference(const Genotype &genotype0, const Genotype &genotype1, float weightFactor, float disjointFactor, const std::unordered_map<FunctionPair, float, FunctionPair> &functionFactors) {
	float difference = 0.0f;

	for (size_t i = 0; i < genotype0._outputNodeIDs.size(); i++)
	for (size_t j = 0; j < genotype1._outputNodeIDs.size(); j++) {
		std::unordered_set<size_t> visitedNodeIDs;
		difference += getDifference(genotype0, genotype1, genotype0._outputNodeIDs[i], genotype1._outputNodeIDs[j], -1, 1.0f, weightFactor, disjointFactor, functionFactors, visitedNodeIDs);
	}

	return difference;
}

void Genotype::removeInput(size_t index) {
	_inputNodeIDs.erase(_inputNodeIDs.begin() + index);
}

void Genotype::removeOutput(size_t index) {
	_outputNodeIDs.erase(_outputNodeIDs.begin() + index);
}

void Genotype::addInputFeedForward(float minWeight, float maxWeight, const std::vector<float> &functionChances, std::mt19937 &generator) {
	// Add a node for this input and fully connect to outputs
	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	size_t newNodeID = _nextNodeID;
	_nextNodeID++;

	std::shared_ptr<Node> newNode = std::make_shared<Node>();

	newNode->_bias = weightDist(generator);
	newNode->_functionIndex = roulette(functionChances, generator);

	for (size_t i = 0; i < _outputNodeIDs.size(); i++) {
		std::shared_ptr<Node> outputNode = _nodes[_outputNodeIDs[i]];

		outputNode->_connections[newNodeID] = weightDist(generator);

		newNode->_outputNodes.insert(_outputNodeIDs[i]);
	}

	_nodes[newNodeID] = newNode;

	_inputNodeIDs.push_back(newNodeID);
}

void Genotype::addOutputFeedForward(float minWeight, float maxWeight, const std::vector<float> &functionChances, std::mt19937 &generator) {
	// Add a node for this output and fully connect to inputs
	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	size_t newNodeID = _nextNodeID;
	_nextNodeID++;

	std::shared_ptr<Node> newNode = std::make_shared<Node>();

	newNode->_bias = weightDist(generator);
	newNode->_functionIndex = roulette(functionChances, generator);

	for (size_t i = 0; i < _inputNodeIDs.size(); i++) {
		std::shared_ptr<Node> inputNode = _nodes[_inputNodeIDs[i]];

		newNode->_connections[_inputNodeIDs[i]] = weightDist(generator);

		inputNode->_outputNodes.insert(newNodeID);
	}

	_nodes[newNodeID] = newNode;

	_outputNodeIDs.push_back(newNodeID);
}

void Genotype::setNumInputsFeedForward(size_t numInputs, float minWeight, float maxWeight, const std::vector<float> &functionChances, std::mt19937 &generator, RemoveMethod removalMethod) {
	switch (removalMethod) {
	case _random:
		while (numInputs < getNumInputs()) {
			std::uniform_int_distribution<int> indexDist(0, getNumInputs() - 1);

			size_t removeIndex = static_cast<size_t>(indexDist(generator));

			removeInput(removeIndex);
		}

		break;

	case _last:
		while (numInputs < getNumInputs())
			removeInput(getNumInputs() - 1);
	}

	while (numInputs > getNumInputs())
		addInputFeedForward(minWeight, maxWeight, functionChances, generator);
}

void Genotype::setNumOutputsFeedForward(size_t numOutputs, float minWeight, float maxWeight, const std::vector<float> &functionChances, std::mt19937 &generator, RemoveMethod removalMethod) {
	switch (removalMethod) {
	case _random:
		while (numOutputs < getNumOutputs()) {
			std::uniform_int_distribution<int> indexDist(0, getNumOutputs() - 1);

			size_t removeIndex = static_cast<size_t>(indexDist(generator));

			removeOutput(removeIndex);
		}

		break;

	case _last:
		while (numOutputs < getNumOutputs())
			removeOutput(getNumOutputs() - 1);

		break;
	}

	while (numOutputs > getNumOutputs())
		addOutputFeedForward(minWeight, maxWeight, functionChances, generator);
}

void Genotype::calculateOutgoingConnections() {
	// Calculate outgoing connections
	for (std::unordered_map<size_t, std::shared_ptr<Node>>::iterator it0 = _nodes.begin(); it0 != _nodes.end(); it0++)
		it0->second->_outputNodes.clear();

	for (std::unordered_map<size_t, std::shared_ptr<Node>>::iterator it0 = _nodes.begin(); it0 != _nodes.end(); it0++)
	for (std::unordered_map<size_t, float>::iterator it1 = it0->second->_connections.begin(); it1 != it0->second->_connections.end();) {
		std::unordered_map<size_t, std::shared_ptr<Node>>::iterator it2 = _nodes.find(it1->first);

		if (it2 == _nodes.end())
			it1 = it0->second->_connections.erase(it1);
		else {
			it2->second->_outputNodes.insert(it0->first);

			it1++;
		}
	}
}

void Genotype::readFromStream(std::istream &is) {
	_nodes.clear();

	_nextNodeID = 0;

	int numNodes;

	is >> numNodes;

	for (int i = 0; i < numNodes; i++) {
		int nodeID;
		std::shared_ptr<Node> newNode = std::make_shared<Node>();

		int numConnections;

		is >> nodeID >> newNode->_bias >> newNode->_functionIndex >> numConnections;

		for (int j = 0; j < numConnections; j++) {
			int connectionID;
			float weight;

			is >> connectionID >> weight;

			newNode->_connections[connectionID] = weight;
		}

		_nodes[nodeID] = newNode;

		_nextNodeID = std::max<int>(_nextNodeID, nodeID) + 1;
	}

	int numInputNodes;

	is >> numInputNodes;

	_inputNodeIDs.resize(numInputNodes);

	for (int i = 0; i < numInputNodes; i++)
		is >> _inputNodeIDs[i];

	int numOutputNodes;

	is >> numOutputNodes;

	_outputNodeIDs.resize(numOutputNodes);

	for (int i = 0; i < numOutputNodes; i++)
		is >> _outputNodeIDs[i];

	// Add output node IDs
	/*for (std::unordered_map<size_t, std::shared_ptr<Node>>::iterator it0 = _nodes.begin(); it0 != _nodes.end(); it0++)
	for (std::unordered_map<size_t, float>::iterator it1 = it0->second->_connections.begin(); it1 != it0->second->_connections.end(); it1++)
		_nodes[it1->first]->_outputNodes.insert(it0->first);*/

	calculateOutgoingConnections();
}

void Genotype::writeToStream(std::ostream &os) const {
	os << _nodes.size() << std::endl;

	for (std::unordered_map<size_t, std::shared_ptr<Node>>::const_iterator cit0 = _nodes.begin(); cit0 != _nodes.end(); cit0++) {
		os << cit0->first << " " << cit0->second->_bias << " " << cit0->second->_functionIndex << " " << cit0->second->_connections.size() << " ";

		for (std::unordered_map<size_t, float>::const_iterator cit1 = cit0->second->_connections.begin(); cit1 != cit0->second->_connections.end(); cit1++) {
			os << cit1->first << " " << cit1->second << " ";
		}

		os << std::endl;
	}

	os << _inputNodeIDs.size() << std::endl;

	for (size_t i = 0; i < _inputNodeIDs.size(); i++)
		os << _inputNodeIDs[i] << " ";

	os << std::endl;

	os << _outputNodeIDs.size() << std::endl;

	for (size_t i = 0; i < _outputNodeIDs.size(); i++)
		os << _outputNodeIDs[i] << " ";

	os << std::endl;
}