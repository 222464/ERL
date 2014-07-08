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

#include <neat/NetworkGenotype.h>

#include <neat/Evolver.h>

#include <neat/UtilFuncs.h>

#include <fstream>
#include <iostream>

#include <unordered_set>

#include <algorithm>

#include <assert.h>

using namespace neat;

NetworkGenotype::ConnectionSet::ConnectionSet()
{}

NetworkGenotype::ConnectionSet::ConnectionSet(const ConnectionSet &other) {
	const size_t numOtherConnections = other._connections.size();
	const size_t numOtherNodes = other._nodes.size();

	_connections.reserve(numOtherConnections);
	_nodes.reserve(numOtherNodes);

	for (size_t i = 0; i < numOtherConnections; i++)
		_connections.push_back(std::shared_ptr<ConnectionGene>(new ConnectionGene(*other._connections[i])));

	for (size_t i = 0; i < numOtherNodes; i++)
		_nodes.push_back(other._nodes[i]);
}

NetworkGenotype::ConnectionSet &NetworkGenotype::ConnectionSet::operator=(const NetworkGenotype::ConnectionSet &other) {
	removeConnections();

	const size_t numOtherConnections = other._connections.size();
	const size_t numOtherNodes = other._nodes.size();

	_connections.reserve(numOtherConnections);
	_nodes.reserve(numOtherNodes);

	for (size_t i = 0; i < numOtherConnections; i++)
		_connections.push_back(std::shared_ptr<ConnectionGene>(new ConnectionGene(*other._connections[i])));

	for (size_t i = 0; i < numOtherNodes; i++)
		_nodes.push_back(other._nodes[i]);

	return *this;
}

void NetworkGenotype::ConnectionSet::addConnection(float minBias, float maxBias, const std::vector<float> &functionChances, const std::shared_ptr<ConnectionGene> &connection, InnovationNumberType &innovationNumber, std::mt19937 &generator) {
	std::uniform_real_distribution<float> biasDist(minBias, maxBias);

	_connections.push_back(connection);

	assert(connection->_inIndex >= 0);

	// Automatically add nodes as needed
	if (connection->_inIndex >= getNumNodes())
		setNumNodes(connection->_inIndex + 1, minBias, maxBias, functionChances, generator);

	assert(connection->_outIndex >= 0);

	// Automatically add nodes as needed. Not initialized until it has connections
	if (connection->_outIndex >= getNumNodes())
		setNumNodes(connection->_outIndex + 1, minBias, maxBias, functionChances, generator);

	// Node initialization: Initialize biases and innovation numbers if not already done so (no prior connections)
	if (_nodes[connection->_inIndex]._connections.empty()) {
		_nodes[connection->_inIndex]._bias = biasDist(generator);
		_nodes[connection->_inIndex]._innovationNumber = innovationNumber;

		innovationNumber++;
	}

	if (_nodes[connection->_outIndex]._connections.empty()) {
		_nodes[connection->_outIndex]._bias = biasDist(generator);
		_nodes[connection->_outIndex]._innovationNumber = innovationNumber;

		innovationNumber++;
	}

	// Add connections
	_nodes[connection->_inIndex]._connections.push_back(connection);
	_nodes[connection->_outIndex]._connections.push_back(connection);
}

void NetworkGenotype::ConnectionSet::addConnectionKnown(float bias1, float bias2, int function1, int function2, float minBias, float maxBias, const std::vector<float> &functionChances, const std::shared_ptr<ConnectionGene> &connection, InnovationNumberType innovationNumber1, InnovationNumberType innovationNumber2, std::mt19937 &generator) {
	_connections.push_back(connection);

	assert(connection->_inIndex >= 0);

	// Automatically add nodes as needed
	if (connection->_inIndex >= getNumNodes())
		setNumNodes(connection->_inIndex + 1, minBias, maxBias, functionChances, generator);

	assert(connection->_outIndex >= 0);

	// Automatically add nodes as needed. Not initialized until it has connections
	if (connection->_outIndex >= getNumNodes())
		setNumNodes(connection->_outIndex + 1, minBias, maxBias, functionChances, generator);

	// Node initialization: Initialize biases and innovation numbers if not already done so (no prior connections)
	if (_nodes[connection->_inIndex]._connections.empty()) {
		_nodes[connection->_inIndex]._bias = bias1;
		_nodes[connection->_inIndex]._innovationNumber = innovationNumber1;
	}

	if (_nodes[connection->_outIndex]._connections.empty()) {
		_nodes[connection->_outIndex]._bias = bias2;
		_nodes[connection->_outIndex]._innovationNumber = innovationNumber2;
	}

	// Add connections
	_nodes[connection->_inIndex]._connections.push_back(connection);
	_nodes[connection->_outIndex]._connections.push_back(connection);
}

void NetworkGenotype::ConnectionSet::removeConnections() {
	_connections.clear();
	_nodes.clear();
}

void NetworkGenotype::removeUnusedNodes() {
	assert(_connectionSet._nodes.size() >= _numOutputs);

	size_t lastHiddenNodeIndex = _connectionSet._nodes.size() - _numOutputs;

	for (size_t i = _numInputs; i < lastHiddenNodeIndex;) {
		bool remove = false;
		bool restart = false;

		if (_connectionSet._nodes[i]._connections.size() >= 2) {
			bool hasInput = false;
			bool hasOutput = false;

			std::list<std::shared_ptr<ConnectionGene>> &connections = _connectionSet._nodes[i]._connections;

			for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = connections.begin(); it != connections.end(); it++) {
				if ((*it)->_outIndex == i)
					hasInput = true;

				if ((*it)->_inIndex == i)
					hasOutput = true;

				if (hasInput && hasOutput)
					break;
			}

			if (!hasInput || !hasOutput) {
				// Remove any connections to the node TODO: NEED TO REMOVE REFERENCES OTHER NODES HAVE TO IT AS WELL!
				/*for(size_t j = 0; j < _connectionSet._connections.size();)
				{
				if(_connectionSet._connections[j]->_inIndex == i || _connectionSet._connections[j]->_outIndex == i)
				{
				delete _connectionSet._connections[j];

				_connectionSet._connections.erase(_connectionSet._connections.begin() + j);

				// Restart search, since more nodes may be removable now
				restart = true;
				}
				else
				j++;
				}*/

				for (size_t j = 0; j < _connectionSet._connections.size(); j++) {
					if (_connectionSet._connections[j]->_inIndex == i || _connectionSet._connections[j]->_outIndex == i) {
						_connectionSet._connections[j]->_enabled = false;

						// Restart search, since more nodes may be removable now
						restart = true;
					}
				}

				remove = true;
			}
			else
				break;
		}
		else
			remove = true;

		if (remove) {
			// Shift indices
			for (size_t j = 0; j < _connectionSet._connections.size(); j++) {
				if (_connectionSet._connections[j]->_inIndex > i)
					_connectionSet._connections[j]->_inIndex--;

				if (_connectionSet._connections[j]->_outIndex > i)
					_connectionSet._connections[j]->_outIndex--;
			}

			_connectionSet._nodes.erase(_connectionSet._nodes.begin() + i);

			assert(_connectionSet._nodes.size() >= _numOutputs);

			lastHiddenNodeIndex = _connectionSet._nodes.size() - _numOutputs;
		}
		else
			i++;

		if (restart)
			i = _numInputs;
	}
}

int NetworkGenotype::rouletteSelectIndex(const std::vector<float> &functionChances, std::mt19937 &generator) {
	float chanceSum = 0.0f;

	for (size_t i = 0; i < functionChances.size(); i++)
		chanceSum += functionChances[i];

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	float randomCusp = dist01(generator) * chanceSum;

	float sumSoFar = 0.0f;

	for (size_t i = 0; i < functionChances.size(); i++) {
		sumSoFar += functionChances[i];

		if (sumSoFar > randomCusp)
			return i;
	}

	return 0;
}

void NetworkGenotype::ConnectionSet::addNodes(int numNodes, float minBias, float maxBias, const std::vector<float> &functionChances, std::mt19937 &generator) {
	assert(_nodes.size() + numNodes >= 0);

	size_t originalSize = _nodes.size();

	_nodes.resize(_nodes.size() + numNodes);

	std::uniform_real_distribution<float> biasDist(minBias, maxBias);

	for (size_t i = originalSize; i < _nodes.size(); i++) {
		_nodes[i]._bias = biasDist(generator);

		_nodes[i]._activationFunctionIndex = rouletteSelectIndex(functionChances, generator);
	}
}

void NetworkGenotype::ConnectionSet::setNumNodes(size_t numNodes, float minBias, float maxBias, const std::vector<float> &functionChances, std::mt19937 &generator) {
	size_t originalSize = _nodes.size();

	_nodes.resize(numNodes);

	if (numNodes > originalSize) {
		std::uniform_real_distribution<float> biasDist(minBias, maxBias);

		for (size_t i = originalSize; i < _nodes.size(); i++) {
			_nodes[i]._bias = biasDist(generator);

			_nodes[i]._activationFunctionIndex = rouletteSelectIndex(functionChances, generator);
		}
	}
}

bool NetworkGenotype::ConnectionSet::canSeverWithoutOrhpan(const ConnectionGene &connection) const {
	if (_nodes[connection._inIndex]._connections.size() == 1)
		return false;

	if (_nodes[connection._outIndex]._connections.size() == 1)
		return false;

	// ------------------------ Input End ------------------------

	{
		size_t numOutputsFromNode = 0;

		const std::list<std::shared_ptr<ConnectionGene>> &inputNodeConnections = _nodes[connection._inIndex]._connections;

		for (std::list<std::shared_ptr<ConnectionGene>>::const_iterator it = inputNodeConnections.begin(); it != inputNodeConnections.end(); it++) {
			if ((*it)->_inIndex == connection._inIndex)
				numOutputsFromNode++;
		}

		if (numOutputsFromNode < 2) // This is the only input to that node
			return false;
	}

	// ------------------------ Output End ------------------------

	{
		size_t numInputsFromNode = 0;

		const std::list<std::shared_ptr<ConnectionGene>> &inputNodeConnections = _nodes[connection._outIndex]._connections;

		for (std::list<std::shared_ptr<ConnectionGene>>::const_iterator it = inputNodeConnections.begin(); it != inputNodeConnections.end(); it++) {
			if ((*it)->_outIndex == connection._outIndex)
				numInputsFromNode++;
		}

		if (numInputsFromNode < 2) // This is the only output to that node
			return false;
	}

	return true;
}

NetworkGenotype::NetworkGenotype()
: _allowSelfConnections(false),
_numInputs(0) // Used to see if was initialized
{}

void NetworkGenotype::initialize(size_t numInputs, size_t numOutputs, float minWeight, float maxWeight, float minBias, float maxBias, const std::vector<float> &functionChances, InnovationNumberType &innovationNumber, std::mt19937 &generator) {
	assert(numInputs > 0);
	assert(numOutputs > 0);

	_connectionSet.removeConnections();

	_numInputs = numInputs;
	_numHidden = 0;
	_numOutputs = numOutputs;

	_connectionSet.setNumNodes(_numInputs + _numOutputs, minBias, maxBias, functionChances, generator);

	std::uniform_real_distribution<float> biasDist(minBias, maxBias);

	for (size_t i = 0; i < _connectionSet.getNumNodes(); i++)
		_connectionSet._nodes[i]._bias = biasDist(generator);

	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	// Completely connect all inputs and outputs directly
	for (size_t i = 0; i < _numInputs; i++)
	for (size_t j = 0; j < _numOutputs; j++) {
		std::shared_ptr<ConnectionGene> connection(new ConnectionGene());
		connection->_enabled = true;
		connection->_inIndex = i;
		connection->_outIndex = _numInputs + j; // Indices of the outputs start at _numInputs

		connection->_weight = weightDist(generator);

		connection->_innovationNumber = innovationNumber;

		innovationNumber++;

		_connectionSet.addConnection(minBias, maxBias, functionChances, connection, innovationNumber, generator);
	}
}

void NetworkGenotype::updateNumHiddenNeurons() {
	assert(_numInputs > 0); // Check to make sure that the genotype was created

	// Get total number of neurons by parsing connections and counting unique connection index occurences
	//std::unordered_set<size_t> indexOccurences;
	size_t largestIndex = 0;

	for (size_t i = 0, size = _connectionSet._connections.size(); i < size; i++) {
		if (_connectionSet._connections[i]->_inIndex > largestIndex)
			largestIndex = _connectionSet._connections[i]->_inIndex;

		if (_connectionSet._connections[i]->_outIndex > largestIndex)
			largestIndex = _connectionSet._connections[i]->_outIndex;
	}

	largestIndex++;

	const size_t numImmutableUnits = _numInputs + _numOutputs;

	if (largestIndex < numImmutableUnits)
		_numHidden = 0;
	else
		_numHidden = largestIndex - numImmutableUnits;
}

void NetworkGenotype::mutatePerturbWeight(float perturbationChance, float maxPerturbation, std::mt19937 &generator) {
	assert(_numInputs > 0); // Check to make sure that the genotype was created

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::uniform_real_distribution<float> perturbationDist(-maxPerturbation, maxPerturbation);

	for (size_t i = 0, size = _connectionSet._connections.size(); i < size; i++)
	if (dist01(generator) < perturbationChance)
		_connectionSet._connections[i]->_weight += perturbationDist(generator);

	// Biases
	for (size_t i = 0, size = _connectionSet._nodes.size(); i < size; i++)
	if (dist01(generator) < perturbationChance)
		_connectionSet._nodes[i]._bias += perturbationDist(generator);
}

void NetworkGenotype::mutatePerturbWeightClamped(float perturbationChance, float maxPerturbation, float minWeight, float maxWeight, float minBias, float maxBias, std::mt19937 &generator) {
	assert(_numInputs > 0); // Check to make sure that the genotype was created

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::uniform_real_distribution<float> perturbationDist(-maxPerturbation, maxPerturbation);

	for (size_t i = 0, size = _connectionSet._connections.size(); i < size; i++)
	if (dist01(generator) < perturbationChance)
		_connectionSet._connections[i]->_weight = std::min(maxWeight, std::max(minWeight, _connectionSet._connections[i]->_weight));

	// Biases
	for (size_t i = 0, size = _connectionSet._nodes.size(); i < size; i++)
	if (dist01(generator) < perturbationChance)
		_connectionSet._nodes[i]._bias = std::min(maxBias, std::max(minBias, _connectionSet._nodes[i]._bias));
}

void NetworkGenotype::mutateChangeFunction(float changeChance, const std::vector<float> &functionChances, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (size_t i = 0; i < _connectionSet._nodes.size(); i++) {
		if (dist01(generator) < changeChance)
			_connectionSet._nodes[i]._activationFunctionIndex = rouletteSelectIndex(functionChances, generator);
	}
}

bool NetworkGenotype::mutateAddConnection(float minWeight, float maxWeight, float minBias, float maxBias, const std::vector<float> &functionChances, InnovationNumberType &innovationNumber, std::mt19937 &generator) {
	assert(_numInputs > 0); // Check to make sure that the genotype was created

	// Takes two unconnected nodes and makes a connection between them.
	// Starts with one node, and sets it as the output of another.
	//assert(_connectionSet.GetNumNodes() > _numInputs);

	// Choose a random node, but skip input nodes. These cannot take inputs
	const size_t numNonInputNodes = _connectionSet.getNumNodes() - _numInputs;

	std::uniform_int_distribution<int> distRandNodeIndex1(_numInputs, _connectionSet.getNumNodes() - 1);

	int randNodeIndex1 = distRandNodeIndex1(generator);

	// If it is connect to every possible node, get a different node
	if (_connectionSet._nodes[randNodeIndex1]._connections.size() >= _connectionSet.getNumNodes()) {
		bool found = false;

		for (size_t offset = 1; offset < numNonInputNodes; offset++) {
			size_t checkIndex = wrap(randNodeIndex1 + offset, _connectionSet.getNumNodes());

			// Skip input nodes
			if (checkIndex < _numInputs)
				checkIndex = _numInputs;

			// Has room for another connection
			if (_connectionSet._nodes[checkIndex]._connections.size() < _connectionSet.getNumNodes()) {
				randNodeIndex1 = checkIndex;

				found = true;

				break;
			}
		}

		// Everything is connected to everything!
		if (!found)
			return false;
	}

	// Choose another random node to connect it to
	std::uniform_int_distribution<int> distRandNodeIndex2(0, _connectionSet.getNumNodes() - 1);

	size_t randNodeIndex2 = distRandNodeIndex2(generator);

	// If connected to self and self connections are not enabled, get new index (increment and wrap)
	if (!_allowSelfConnections && randNodeIndex1 == randNodeIndex2) {
		// Cannot connect to self, increment index
		randNodeIndex2 = wrap(randNodeIndex2 + 1, _connectionSet.getNumNodes());
	}

	// If these nodes are already connected, find a different node
	std::list<std::shared_ptr<ConnectionGene>> &node1List = _connectionSet._nodes[randNodeIndex1]._connections;

	const size_t maxAttempts = _connectionSet.getNumNodes() - 1;
	size_t attempt = 1;

	assert(node1List.size() < _connectionSet.getNumNodes());

	for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = node1List.begin(); it != node1List.end(); it++) {
		// If the connection already exists, find a different node
		if ((*it)->_inIndex == randNodeIndex2) {
			// Try a different node, restart checks
			assert(attempt++ <= maxAttempts);

			// Have to make sure that no attempt is being made to connect to self again
			if (!_allowSelfConnections && randNodeIndex1 == randNodeIndex2 + 1) {
				// Cannot connect to self, increment index
				randNodeIndex2 = wrap(randNodeIndex2 + 2, _connectionSet.getNumNodes());
			}
			else
				randNodeIndex2 = wrap(randNodeIndex2 + 1, _connectionSet.getNumNodes());

			// Restart checks for new node
			it = node1List.begin();
		}
	}

	// Create the connection
	std::shared_ptr<ConnectionGene> connection(new ConnectionGene());
	connection->_enabled = true;
	connection->_outIndex = randNodeIndex1;
	connection->_inIndex = randNodeIndex2;

	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	connection->_weight = weightDist(generator);

	connection->_innovationNumber = innovationNumber;

	innovationNumber++;

	_connectionSet.addConnection(minBias, maxBias, functionChances, connection, innovationNumber, generator);

	return true;
}

void NetworkGenotype::mutateAddNode(float minWeight, float maxWeight, float minBias, float maxBias, const std::vector<float> &functionChances, InnovationNumberType &innovationNumber, std::mt19937 &generator) {
	assert(_numInputs > 0); // Check to make sure that the genotype was created

	_numHidden++;

	const size_t newNodeIndex = _connectionSet.getNumNodes(); // New node will be at back of vector

	// Take random connection, split it, and insert the new node

	// If there are no connections to split, something went wrong
	if (_connectionSet.getNumConnections() == 0)
		connectUnconnectedOutputs(minWeight, maxWeight, minBias, maxBias, functionChances, innovationNumber, generator);

	std::uniform_int_distribution<int> distConnection(0, _connectionSet.getNumConnections() - 1);

	int randConnectionIndex = distConnection(generator);

	std::shared_ptr<ConnectionGene> geneToBeSplit = _connectionSet._connections[randConnectionIndex];

	// Disable this connection (may be re-enabled in crossover)
	geneToBeSplit->_enabled = false;

	// First new connection (not reusing the old one)
	std::shared_ptr<ConnectionGene> connection1(new ConnectionGene());

	connection1->_enabled = true;
	connection1->_inIndex = geneToBeSplit->_inIndex;
	connection1->_outIndex = newNodeIndex;

	std::uniform_real_distribution<float> distWeight(minWeight, maxWeight);

	connection1->_weight = distWeight(generator);

	connection1->_innovationNumber = innovationNumber;

	innovationNumber++;

	_connectionSet.addConnection(minBias, maxBias, functionChances, connection1, innovationNumber, generator);

	// Second new connection (not reusing the old one)
	std::shared_ptr<ConnectionGene> connection2(new ConnectionGene());

	connection2->_enabled = true;
	connection2->_inIndex = newNodeIndex;
	connection2->_outIndex = geneToBeSplit->_outIndex;

	connection2->_weight = distWeight(generator);

	connection2->_innovationNumber = innovationNumber;

	innovationNumber++;

	_connectionSet.addConnection(minBias, maxBias, functionChances, connection2, innovationNumber, generator);
}

void NetworkGenotype::crossover(const NetworkGenotype &otherParent, NetworkGenotype &child, float disableGeneChance, float fitnessForThis, float fitnessForOtherParent, float minBias, float maxBias, const std::vector<float> &functionChances, std::mt19937 &generator) {
#ifdef DEBUG
	for (size_t i = 0, numConnections = _connectionSet.GetNumConnections(); i < numConnections; i++)
		assert(_connectionSet._connections[i]->_outIndex >= _numInputs);

	for(size_t i = 0, numConnections = otherParent._connectionSet.GetNumConnections(); i < numConnections; i++)
		assert(otherParent._connectionSet._connections[i]->_outIndex >= otherParent._numInputs);
#endif

	// Compile pointers to genes with identical historical markers into 2 lists
	struct GenePair {
		std::shared_ptr<ConnectionGene> _gene1, _gene2;
	};

	std::list<GenePair> matchingHistoryConnectionGenes;

	// Additional list for those that do not match (disjoint or excess)
	std::list<std::shared_ptr<ConnectionGene>> notMatchingHistoryConnectionGenes1;

	// Copy gene pointers of second gene into linked list so can remove easily.
	// What is left over in this list will be disjoint/excess genes
	std::list<std::shared_ptr<ConnectionGene>> otherParentInnovationNumbers;

	for (size_t i = 0; i < otherParent._connectionSet.getNumConnections(); i++)
		otherParentInnovationNumbers.push_back(otherParent._connectionSet._connections[i]);

	for (size_t i = 0; i < _connectionSet.getNumConnections(); i++) {
		bool matchFound = false;

		for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = otherParentInnovationNumbers.begin(); it != otherParentInnovationNumbers.end();) {
			if (_connectionSet._connections[i]->_innovationNumber == (*it)->_innovationNumber) {
				GenePair newMatchingSet;
				newMatchingSet._gene1 = _connectionSet._connections[i];
				newMatchingSet._gene2 = *it;

				matchingHistoryConnectionGenes.push_back(newMatchingSet);

				// Erase from second list
				it = otherParentInnovationNumbers.erase(it);

				matchFound = true;

				break;
			}
			else
				it++;
		}

		if (!matchFound)
			notMatchingHistoryConnectionGenes1.push_back(_connectionSet._connections[i]);
	}

	// UNIFORM CROSSOVER

	// Create child
	child._numInputs = std::max(_numInputs, otherParent._numInputs);
	child._numOutputs = std::max(_numOutputs, otherParent._numOutputs);

	child._connectionSet.removeConnections(); // Clear existing connections

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (std::list<GenePair>::iterator it = matchingHistoryConnectionGenes.begin(); it != matchingHistoryConnectionGenes.end(); it++) {
		std::shared_ptr<ConnectionGene> connection;

		const NetworkGenotype* pSelected;

		if (it->_gene1->_outIndex < child._numInputs) {
			pSelected = &otherParent;
			connection.reset(new ConnectionGene(*(it->_gene2)));
		}
		else if (it->_gene2->_outIndex < child._numInputs) {
			pSelected = this;
			connection.reset(new ConnectionGene(*(it->_gene1)));
		}
		else if (dist01(generator) < 0.5f) {
			pSelected = this;
			connection.reset(new ConnectionGene(*(it->_gene1)));
		}
		else {
			pSelected = &otherParent;
			connection.reset(new ConnectionGene(*(it->_gene2)));
		}

		child._connectionSet.addConnectionKnown(pSelected->_connectionSet._nodes[connection->_inIndex]._bias,
			pSelected->_connectionSet._nodes[connection->_outIndex]._bias, 
			pSelected->_connectionSet._nodes[connection->_inIndex]._activationFunctionIndex,
			pSelected->_connectionSet._nodes[connection->_outIndex]._activationFunctionIndex,
			minBias, maxBias, functionChances,
			connection,
			pSelected->_connectionSet._nodes[connection->_inIndex]._innovationNumber,
			pSelected->_connectionSet._nodes[connection->_outIndex]._innovationNumber,
			generator);

		// Random disable/enable
		if ((!it->_gene1->_enabled || !it->_gene2->_enabled) && dist01(generator) < disableGeneChance)
			connection->_enabled = false;
		else
			connection->_enabled = true;
	}

	// notMatchingHistoryConnectionGenes1 and otherParentInnovationNumbers hold disjoint/excess genes from this or other parent respectively now
	// Give child the disjoint genes from the more fit parent
	// If they have the same fitness, take some randomly from both parents
	enum SelectionMode {
		_this, _other, _both
	} selectionMode = _both;

	const size_t minNumConnections = std::min<size_t>(_connectionSet.getNumConnections(), otherParent._connectionSet.getNumConnections());

	if (fitnessForThis > fitnessForOtherParent) {
		selectionMode = _this;

		for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = notMatchingHistoryConnectionGenes1.begin(); it != notMatchingHistoryConnectionGenes1.end(); it++) {
			if ((*it)->_outIndex < child._numInputs)
				continue;

			std::shared_ptr<ConnectionGene> connection(new ConnectionGene(*(*it)));

			child._connectionSet.addConnectionKnown(_connectionSet._nodes[connection->_inIndex]._bias,
				_connectionSet._nodes[connection->_outIndex]._bias, 
				_connectionSet._nodes[connection->_inIndex]._activationFunctionIndex,
				_connectionSet._nodes[connection->_outIndex]._activationFunctionIndex,
				minBias, maxBias, functionChances,
				connection,
				_connectionSet._nodes[connection->_inIndex]._innovationNumber,
				_connectionSet._nodes[connection->_outIndex]._innovationNumber,
				generator);

#ifdef DISABLE_CHANCE_EXCESS_DISJOINT
			if (!connection->_enabled && dist01(generator) < disableGeneChance)
				connection->_enabled = false;
			else
				connection->_enabled = true;
#endif
		}
	}
	else if (fitnessForThis < fitnessForOtherParent) {
		selectionMode = _other;

		for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = otherParentInnovationNumbers.begin(); it != otherParentInnovationNumbers.end(); it++) {
			if ((*it)->_outIndex < child._numInputs)
				continue;

			std::shared_ptr<ConnectionGene> connection(new ConnectionGene(*(*it)));

			child._connectionSet.addConnectionKnown(otherParent._connectionSet._nodes[connection->_inIndex]._bias,
				otherParent._connectionSet._nodes[connection->_outIndex]._bias, 
				otherParent._connectionSet._nodes[connection->_inIndex]._activationFunctionIndex,
				otherParent._connectionSet._nodes[connection->_outIndex]._activationFunctionIndex,
				minBias, maxBias, functionChances,
				connection,
				otherParent._connectionSet._nodes[connection->_inIndex]._innovationNumber,
				otherParent._connectionSet._nodes[connection->_inIndex]._innovationNumber,
				generator);

#ifdef DISABLE_CHANCE_EXCESS_DISJOINT
			if (!connection->_enabled && dist01(generator) < disableGeneChance)
				connection->_enabled = false;
			else
				connection->_enabled = true;
#endif
		}
	}
	else if (!notMatchingHistoryConnectionGenes1.empty() || !otherParentInnovationNumbers.empty()) {
		selectionMode = _both;

		// Number of disjoint/excess genes inherited is average of number of genes from each
		size_t numInherit;

		if (minNumConnections > child._connectionSet.getNumConnections())
			numInherit = std::max<size_t>(minNumConnections - child._connectionSet.getNumConnections(),
			static_cast<size_t>(static_cast<float>(notMatchingHistoryConnectionGenes1.size() + otherParentInnovationNumbers.size()) * 0.5f));
		else
			numInherit = static_cast<size_t>(static_cast<float>(notMatchingHistoryConnectionGenes1.size() + otherParentInnovationNumbers.size()) * 0.5f);

		for (size_t i = 0; i < numInherit; i++) {
			// Alternate
			if ((dist01(generator) < 0.5f && !notMatchingHistoryConnectionGenes1.empty()) || otherParentInnovationNumbers.empty()) {
				std::uniform_int_distribution<int> distNotMatchingHistoryConnectionGenes1(0, notMatchingHistoryConnectionGenes1.size() - 1);

				int randIndex = distNotMatchingHistoryConnectionGenes1(generator);

				// Get pointer by advancing iterator
				std::list<std::shared_ptr<ConnectionGene>>::iterator it = notMatchingHistoryConnectionGenes1.begin();

				for (size_t j = 0; j < randIndex; j++, it++) {
					if ((*it)->_outIndex < child._numInputs) {
						j++;
						it++;
					}
				}

				std::shared_ptr<ConnectionGene> connection(new ConnectionGene(*(*it)));

				notMatchingHistoryConnectionGenes1.erase(it);

				child._connectionSet.addConnectionKnown(_connectionSet._nodes[connection->_inIndex]._bias,
					_connectionSet._nodes[connection->_outIndex]._bias, 
					_connectionSet._nodes[connection->_inIndex]._activationFunctionIndex,
					_connectionSet._nodes[connection->_outIndex]._activationFunctionIndex,
					minBias, maxBias, functionChances,
					connection,
					_connectionSet._nodes[connection->_inIndex]._innovationNumber,
					_connectionSet._nodes[connection->_inIndex]._innovationNumber,
					generator);

#ifdef DISABLE_CHANCE_EXCESS_DISJOINT
				if (!connection->_enabled && dist01(generator) < disableGeneChance)
					connection->_enabled = false;
				else
					connection->_enabled = true;
#endif
			}
			else {
				std::uniform_int_distribution<int> distOtherParentInnovationNumbers(0, otherParentInnovationNumbers.size() - 1);

				int randIndex = distOtherParentInnovationNumbers(generator);

				// Get pointer by advancing iterator
				std::list<std::shared_ptr<ConnectionGene>>::iterator it = otherParentInnovationNumbers.begin();

				for (size_t j = 0; j < randIndex; j++, it++) {
					if ((*it)->_outIndex < child._numInputs) {
						j++;
						it++;
					}
				}

				std::shared_ptr<ConnectionGene> connection(new ConnectionGene(*(*it)));

				otherParentInnovationNumbers.erase(it);

				child._connectionSet.addConnectionKnown(otherParent._connectionSet._nodes[connection->_inIndex]._bias,
					otherParent._connectionSet._nodes[connection->_outIndex]._bias, 
					otherParent._connectionSet._nodes[connection->_inIndex]._activationFunctionIndex,
					otherParent._connectionSet._nodes[connection->_outIndex]._activationFunctionIndex,
					minBias, maxBias, functionChances,
					connection,
					otherParent._connectionSet._nodes[connection->_inIndex]._innovationNumber,
					otherParent._connectionSet._nodes[connection->_inIndex]._innovationNumber,
					generator);

#ifdef DISABLE_CHANCE_EXCESS_DISJOINT
				if (!connection->_enabled && dist01(generator) < disableGeneChance)
					connection->_enabled = false;
				else
					connection->_enabled = true;
#endif
			}
		}
	}

	// If not enough, inherit from other one after all in addition to these connections
	if (child._connectionSet._connections.size() < minNumConnections) {
		assert(selectionMode != _both);

		if (selectionMode == _this) {
			selectionMode = _other;

			for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = otherParentInnovationNumbers.begin(); it != otherParentInnovationNumbers.end() && child._connectionSet._connections.size() < minNumConnections; it++) {
				if ((*it)->_outIndex < child._numInputs)
					continue;

				std::shared_ptr<ConnectionGene> connection(new ConnectionGene(*(*it)));

				child._connectionSet.addConnectionKnown(otherParent._connectionSet._nodes[connection->_inIndex]._bias,
					otherParent._connectionSet._nodes[connection->_outIndex]._bias, 
					otherParent._connectionSet._nodes[connection->_inIndex]._activationFunctionIndex,
					otherParent._connectionSet._nodes[connection->_outIndex]._activationFunctionIndex,
					minBias, maxBias, functionChances,
					connection,
					otherParent._connectionSet._nodes[connection->_inIndex]._innovationNumber,
					otherParent._connectionSet._nodes[connection->_inIndex]._innovationNumber,
					generator);

#ifdef DISABLE_CHANCE_EXCESS_DISJOINT
				if (!connection->_enabled && dist01(generator) < disableGeneChance)
					connection->_enabled = false;
				else
					connection->_enabled = true;
#endif
			}
		}
		else {
			for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = notMatchingHistoryConnectionGenes1.begin(); it != notMatchingHistoryConnectionGenes1.end() && child._connectionSet._connections.size() < minNumConnections; it++) {
				if ((*it)->_outIndex < child._numInputs)
					continue;

				std::shared_ptr<ConnectionGene> connection(new ConnectionGene(*(*it)));

				child._connectionSet.addConnectionKnown(_connectionSet._nodes[connection->_inIndex]._bias,
					_connectionSet._nodes[connection->_outIndex]._bias, 
					_connectionSet._nodes[connection->_inIndex]._activationFunctionIndex,
					_connectionSet._nodes[connection->_outIndex]._activationFunctionIndex,
					minBias, maxBias, functionChances,
					connection,
					_connectionSet._nodes[connection->_inIndex]._innovationNumber,
					_connectionSet._nodes[connection->_outIndex]._innovationNumber,
					generator);

#ifdef DISABLE_CHANCE_EXCESS_DISJOINT
				if (!connection->_enabled && dist01(generator) < disableGeneChance)
					connection->_enabled = false;
				else
					connection->_enabled = true;
#endif
			}
		}
	}

	// "Check up" on the disabled genes to make sure they won't create orphans
	for (size_t i = 0, numConnections = child._connectionSet._connections.size(); i < numConnections; i++) {
		if (child._connectionSet._connections[i]->_enabled) {
			// Re-enable if it cannot be severed
			if (!child._connectionSet.canSeverWithoutOrhpan(*child._connectionSet._connections[i]))
				child._connectionSet._connections[i]->_enabled = true;
		}
	}

	// Recalculate number of hidden
	child.updateNumHiddenNeurons();

#ifdef DEBUG
	for (size_t i = 0, numConnections = child._connectionSet.GetNumConnections(); i < numConnections; i++)
		assert(child._connectionSet._connections[i]->_outIndex >= child._numInputs);
#endif
}

float NetworkGenotype::getSimilarity(const NetworkGenotype &other, float excessFactor, float disjointFactor, float averageWeightDifferenceFactor, float inputCountDifferenceFactor, float outputCountDifferenceFactor, float activationFunctionFactor) {
	// Get number of excess and disjoint genes
	float totalWeightDifference = 0.0f;
	float totalActivationFunctionDifference = 0.0f;

	size_t numMatching = 0;

	size_t numDisjointConnectionGenes1 = 0;
	size_t numExcessConnectionGenes1 = 0;

	size_t numDisjointConnectionGenes2 = 0;
	size_t numExcessConnectionGenes2 = 0;

	// Copy gene pointers of second gene into linked list so can remove easily.
	// What is left over in this list will be disjoint/excess genes
	std::list<std::shared_ptr<ConnectionGene>> otherGeneInnovationNumbers;

	InnovationNumberType maxInnovation2 = 0;

	for (size_t i = 0, numConnections_otherParent = other._connectionSet.getNumConnections(); i < numConnections_otherParent; i++) {
		if (other._connectionSet._connections[i]->_innovationNumber > maxInnovation2)
			maxInnovation2 = other._connectionSet._connections[i]->_innovationNumber;

		otherGeneInnovationNumbers.push_back(other._connectionSet._connections[i]);
	}

	// Count max innovation number for first gene
	InnovationNumberType maxInnovation1 = 0;

	for (size_t i = 0, numConnections_this = _connectionSet.getNumConnections(); i < numConnections_this; i++)
	if (_connectionSet._connections[i]->_innovationNumber > maxInnovation1)
		maxInnovation1 = _connectionSet._connections[i]->_innovationNumber;

	for (size_t i = 0, numConnections_this = _connectionSet.getNumConnections(); i < numConnections_this; i++) {
		bool matchFound = false;

		std::list<std::shared_ptr<ConnectionGene>>::iterator it = otherGeneInnovationNumbers.begin();

		if (_connectionSet._connections[i]->_innovationNumber > maxInnovation2) {
			numExcessConnectionGenes2++;

			continue;
		}

		while (it != otherGeneInnovationNumbers.end()) {
			if (_connectionSet._connections[i]->_innovationNumber == (*it)->_innovationNumber) {
				matchFound = true;

				break;
			}
			else
				it++;
		}

		if (matchFound) {
			// Get average weight difference
			totalWeightDifference += std::abs((*it)->_weight - _connectionSet._connections[i]->_weight);

			// Erase from second list
			otherGeneInnovationNumbers.erase(it);

			numMatching++;
		}
		else
			numDisjointConnectionGenes1++;
	}

	// Now add in the biases for the matching genes (node innovation number matching)

	// Build list of unmatched indices so don't repeat search
	std::list<size_t> otherNodeUnmatchedIndices;

	for (size_t i = 0, numNodes2 = other._connectionSet.getNumNodes(); i < numNodes2; i++)
		otherNodeUnmatchedIndices.push_back(i);

	for (size_t i = 0, numNodes1 = _connectionSet.getNumNodes(); i < numNodes1; i++)
	for (std::list<size_t>::iterator it = otherNodeUnmatchedIndices.begin(); it != otherNodeUnmatchedIndices.end(); it++) {
		if (_connectionSet._nodes[i]._innovationNumber == other._connectionSet._nodes[*it]._innovationNumber) {
			totalWeightDifference += fabsf(_connectionSet._nodes[i]._bias - other._connectionSet._nodes[*it]._bias);

			if (_connectionSet._nodes[i]._activationFunctionIndex != other._connectionSet._nodes[*it]._activationFunctionIndex)
				totalActivationFunctionDifference++;

			numMatching++;

			otherNodeUnmatchedIndices.erase(it);

			break;
		}
	}

	// Now count other's excess/disjoint from remains in otherGeneInnovationNumbers
	for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = otherGeneInnovationNumbers.begin(); it != otherGeneInnovationNumbers.end(); it++) {
		if ((*it)->_innovationNumber > maxInnovation1)
			numExcessConnectionGenes1++;
		else
			numDisjointConnectionGenes2++;
	}

#ifdef NORMALIZE_EXCESS_DISJOINT_SIMILARITY
	float normalizationFactor = 1.0f / std::max<size_t>(_connectionSet.getNumConnections(), other._connectionSet.getNumConnections());
#else
	float normalizationFactor = 1.0f;
#endif

	// Calculate similarity with weightings
	if(numMatching == 0) // Avoid / 0
		return excessFactor * static_cast<float>(numExcessConnectionGenes1 + numExcessConnectionGenes2) * normalizationFactor +
			disjointFactor * static_cast<float>(numDisjointConnectionGenes1 + numDisjointConnectionGenes2) * normalizationFactor +
			inputCountDifferenceFactor * std::abs(static_cast<float>(_numInputs) - static_cast<float>(other._numInputs)) +
			outputCountDifferenceFactor * std::abs(static_cast<float>(_numOutputs) - static_cast<float>(other._numOutputs)) +
			activationFunctionFactor * totalActivationFunctionDifference;

	return excessFactor * static_cast<float>(numExcessConnectionGenes1 + numExcessConnectionGenes2) * normalizationFactor +
		disjointFactor * static_cast<float>(numDisjointConnectionGenes1 + numDisjointConnectionGenes2) * normalizationFactor +
		averageWeightDifferenceFactor * (totalWeightDifference / numMatching) +
		inputCountDifferenceFactor * std::abs(static_cast<float>(_numInputs) - static_cast<float>(other._numInputs)) +
		outputCountDifferenceFactor * std::abs(static_cast<float>(_numOutputs) - static_cast<float>(other._numOutputs)) +
		activationFunctionFactor * totalActivationFunctionDifference;
}

void NetworkGenotype::initialize(size_t numInputs, size_t numOutputs, const EvolverSettings* pSettings, const std::vector<float> &functionChances, InnovationNumberType &innovationNumber, std::mt19937 &generator) {
	initialize(numInputs, numOutputs, pSettings->_minWeight, pSettings->_maxWeight, pSettings->_minBias, pSettings->_maxBias, functionChances, innovationNumber, generator);

	updateNumHiddenNeurons();
}

void NetworkGenotype::crossover(const EvolverSettings* pSettings, const std::vector<float> &functionChances, const Evolvable* pOtherParent, Evolvable* pChild, float fitnessForThis, float fitnessForOtherParent, InnovationNumberType &innovationNumber, std::mt19937 &generator) {
	const NetworkGenotype* pOtherParentNetworkGenotype = static_cast<const NetworkGenotype*>(pOtherParent);
	NetworkGenotype* pChildNetworkGenotype = static_cast<NetworkGenotype*>(pChild);
	
	crossover(*pOtherParentNetworkGenotype, *pChildNetworkGenotype, pSettings->_disableGeneChance, fitnessForThis, fitnessForOtherParent, pSettings->_minBias, pSettings->_maxBias, functionChances, generator);

	updateNumHiddenNeurons();
}

void NetworkGenotype::mutate(const EvolverSettings* pSettings, const std::vector<float> &functionChances, InnovationNumberType &innovationNumber, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	if (dist01(generator) < pSettings->_newNodeMutationRate)
		mutateAddNode(pSettings->_minWeight, pSettings->_maxWeight, pSettings->_minBias, pSettings->_maxBias, functionChances, innovationNumber, generator);

	if (dist01(generator) < pSettings->_newConnectionMutationRate)
		mutateAddConnection(pSettings->_minWeight, pSettings->_maxWeight, pSettings->_minBias, pSettings->_maxBias, functionChances, innovationNumber, generator);

	mutatePerturbWeight(pSettings->_weightPerturbationChance, pSettings->_maxPerturbation, generator);

	mutateChangeFunction(pSettings->_changeFunctionChance, functionChances, generator);

	updateNumHiddenNeurons();
}

float NetworkGenotype::getSimilarity(const EvolverSettings* pSettings, const std::vector<float> &functionChances, const Evolvable* pOther) {
	const NetworkGenotype* pOtherNetworkGenotype = static_cast<const NetworkGenotype*>(pOther);
	
	return getSimilarity(*pOtherNetworkGenotype, pSettings->_excessFactor, pSettings->_disjointFactor, pSettings->_averageWeightDifferenceFactor, pSettings->_inputCountDifferenceFactor, pSettings->_outputCountDifferenceFactor, pSettings->_functionFactor);
}

void NetworkGenotype::setNumInputs(size_t numInputs) {
#ifdef DEBUG
	for (size_t i = 0, numConnections = _connectionSet.GetNumConnections(); i < numConnections; i++)
		assert(_connectionSet._connections[i]->_outIndex >= _numInputs);
#endif

	// Shift over all indices by change in inputs
	int dNodes = static_cast<int>(numInputs) - static_cast<int>(_numInputs);

	if (dNodes > 0) {
		for (size_t i = 0, numConnections = _connectionSet.getNumConnections(); i < numConnections; i++) {
			if (_connectionSet._connections[i]->_inIndex >= _numInputs)
				_connectionSet._connections[i]->_inIndex += dNodes;

			//assert(_connectionSet._connections[i]->_outIndex >= _numInputs);

			_connectionSet._connections[i]->_outIndex += dNodes;
		}

		// Insert nodes for the inputs
		_connectionSet._nodes.insert(_connectionSet._nodes.begin() + _numInputs, dNodes, NodeData());
	}
	else if (dNodes < 0) {
		for (size_t i = 0; i < _connectionSet.getNumConnections();) {
			// If it was connected to an input that is now removed, delete the connection
			if (_connectionSet._connections[i]->_inIndex < _numInputs && _connectionSet._connections[i]->_inIndex >= numInputs) {
				// Remove references from nodes it connects
				NodeData &node = _connectionSet._nodes[_connectionSet._connections[i]->_inIndex];

				bool found = false;

				for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = node._connections.begin(); it != node._connections.end(); it++)
				if ((*it) == _connectionSet._connections[i]) {
					node._connections.erase(it);

					found = true;

					break;
				}

				//assert(found);

				found = false;

				node = _connectionSet._nodes[_connectionSet._connections[i]->_outIndex];

				for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = node._connections.begin(); it != node._connections.end(); it++)
				if ((*it) == _connectionSet._connections[i]) {
					node._connections.erase(it);

					found = true;

					break;
				}

				//assert(found);

				_connectionSet._connections.erase(_connectionSet._connections.begin() + i);
			}
			else {
				if (_connectionSet._connections[i]->_inIndex >= _numInputs)
					_connectionSet._connections[i]->_inIndex += dNodes;

				//assert(_connectionSet._connections[i]->_outIndex >= _numInputs);

				_connectionSet._connections[i]->_outIndex += dNodes;

				i++;
			}
		}

		_connectionSet._nodes.erase(_connectionSet._nodes.begin() + numInputs, _connectionSet._nodes.begin() + _numInputs);
	}

	_numInputs = numInputs;

#ifdef DEBUG
	for (size_t i = 0, numConnections = _connectionSet.GetNumConnections(); i < numConnections; i++)
		assert(_connectionSet._connections[i]->_outIndex >= _numInputs);
#endif

	updateNumHiddenNeurons();
}

void NetworkGenotype::setNumOutputs(size_t numOutputs, float minBias, float maxBias, const std::vector<float> &functionChances, std::mt19937 &generator) {
#ifdef DEBUG
	for (size_t i = 0, numConnections = _connectionSet.GetNumConnections(); i < numConnections; i++)
		assert(_connectionSet._connections[i]->_outIndex >= _numInputs);
#endif

	// Shift over all indices by change in inputs
	int dNodes = static_cast<int>(numOutputs) - static_cast<int>(_numOutputs);

	if (dNodes < 0) {
		for (size_t i = 0; i < _connectionSet.getNumConnections();) {
			// If it was connected to an input that is now removed, delete the connection
			if (_connectionSet._connections[i]->_inIndex >= numOutputs || _connectionSet._connections[i]->_outIndex >= numOutputs) {
				// Remove references from nodes it connects
				NodeData &node = _connectionSet._nodes[_connectionSet._connections[i]->_inIndex];

				bool found = false;

				for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = node._connections.begin(); it != node._connections.end(); it++)
				if ((*it) == _connectionSet._connections[i]) {
					node._connections.erase(it);

					found = true;

					break;
				}

				//assert(found);

				found = false;

				node = _connectionSet._nodes[_connectionSet._connections[i]->_outIndex];

				for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = node._connections.begin(); it != node._connections.end(); it++)
				if ((*it) == _connectionSet._connections[i]) {
					node._connections.erase(it);

					found = true;

					break;
				}

				//assert(found);

				_connectionSet._connections.erase(_connectionSet._connections.begin() + i);
			}
			else
				i++;
		}
	}

	_numOutputs = numOutputs;

	_connectionSet.setNumNodes(_numInputs + _numHidden + _numOutputs, minBias, maxBias, functionChances, generator);

#ifdef DEBUG
	for (size_t i = 0, numConnections = _connectionSet.GetNumConnections(); i < numConnections; i++)
		assert(_connectionSet._connections[i]->_outIndex >= _numInputs);
#endif

	updateNumHiddenNeurons();
}

void NetworkGenotype::setNumInputsFullyConnect(size_t numInputs, float minWeight, float maxWeight, float minBias, float maxBias, const std::vector<float> &functionChances, InnovationNumberType &innovationNumber, std::mt19937 &generator) {
#ifdef DEBUG
	for (size_t i = 0, numConnections = _connectionSet.GetNumConnections(); i < numConnections; i++)
		assert(_connectionSet._connections[i]->_outIndex >= _numInputs);
#endif

	// Shift over all indices by change in inputs
	int dNodes = static_cast<int>(numInputs) - static_cast<int>(_numInputs);

	const int newNumNodes = static_cast<int>(numInputs + _numHidden + _numOutputs);

	std::uniform_real_distribution<float> weightDist(minWeight, maxWeight);

	if (dNodes > 0) {
		for (size_t i = 0, numConnections = _connectionSet.getNumConnections(); i < numConnections; i++) {
			if (_connectionSet._connections[i]->_inIndex >= _numInputs)
				_connectionSet._connections[i]->_inIndex += dNodes;

			assert(_connectionSet._connections[i]->_outIndex >= _numInputs);

			_connectionSet._connections[i]->_outIndex += dNodes;
		}

		// Insert nodes for the inputs
		_connectionSet._nodes.insert(_connectionSet._nodes.begin() + _numInputs, dNodes, NodeData());

		// Connect new inputs to all outputs
		for (int i = 0; i < dNodes; i++)
		for (size_t j = 0; j < _numOutputs; j++) {
			std::shared_ptr<ConnectionGene> connection(new ConnectionGene());

			connection->_enabled = true;
			connection->_inIndex = _numInputs + i;

			assert(newNumNodes - 1 - static_cast<signed>(j) >= static_cast<signed>(numInputs));
			connection->_outIndex = newNumNodes - 1 - j;

			connection->_weight = weightDist(generator);

			connection->_innovationNumber = innovationNumber;

			innovationNumber++;

			_connectionSet.addConnection(minBias, maxBias, functionChances, connection, innovationNumber, generator);
		}
	}
	else if (dNodes < 0) {
		for (size_t i = 0; i < _connectionSet.getNumConnections();) {
			// If it was connected to an input that is now removed, delete the connection
			if (_connectionSet._connections[i]->_inIndex < _numInputs && _connectionSet._connections[i]->_inIndex >= numInputs) {
				// Remove references from nodes it connects
				NodeData &node = _connectionSet._nodes[_connectionSet._connections[i]->_inIndex];

				bool found = false;

				for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = node._connections.begin(); it != node._connections.end(); it++)
				if ((*it) == _connectionSet._connections[i]) {
					node._connections.erase(it);

					found = true;

					break;
				}

				//assert(found);

				found = false;

				node = _connectionSet._nodes[_connectionSet._connections[i]->_outIndex];

				for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = node._connections.begin(); it != node._connections.end(); it++)
				if ((*it) == _connectionSet._connections[i]) {
					node._connections.erase(it);

					found = true;

					break;
				}

				//assert(found);

				_connectionSet._connections.erase(_connectionSet._connections.begin() + i);
			}
			else {
				if (_connectionSet._connections[i]->_inIndex >= _numInputs)
					_connectionSet._connections[i]->_inIndex += dNodes;

				assert(_connectionSet._connections[i]->_outIndex >= _numInputs);

				_connectionSet._connections[i]->_outIndex += dNodes;

				i++;
			}
		}

		_connectionSet._nodes.erase(_connectionSet._nodes.begin() + numInputs, _connectionSet._nodes.begin() + _numInputs);
	}

	_numInputs = numInputs;

#ifdef DEBUG
	for (size_t i = 0, numConnections = _connectionSet.GetNumConnections(); i < numConnections; i++)
		assert(_connectionSet._connections[i]->_outIndex >= _numInputs);
#endif

	updateNumHiddenNeurons();
}

void NetworkGenotype::setNumOutputsFullyConnect(size_t numOutputs, float minWeight, float maxWeight, float minBias, float maxBias, const std::vector<float> &functionChances, InnovationNumberType &innovationNumber, std::mt19937 &generator) {
#ifdef DEBUG
	for (size_t i = 0, numConnections = _connectionSet.GetNumConnections(); i < numConnections; i++)
		assert(_connectionSet._connections[i]->_outIndex >= _numInputs);
#endif

	// Shift over all indices by change in inputs
	int dNodes = static_cast<int>(numOutputs) - static_cast<int>(_numOutputs);

	updateNumHiddenNeurons();

	const size_t newNumNodes = _numInputs + _numHidden + numOutputs;

	std::uniform_real_distribution<float> distWeight(minWeight, maxWeight);

	if (dNodes > 0) {
		// Connect new outputs to all inputs
		for (int i = 0; i < dNodes; i++)
		for (size_t j = 0; j < _numInputs; j++) {
			std::shared_ptr<ConnectionGene> connection(new ConnectionGene());

			connection->_enabled = true;
			connection->_inIndex = j;

			assert(static_cast<signed>(newNumNodes)-1 - static_cast<signed>(i) > static_cast<signed>(_numInputs));

			connection->_outIndex = newNumNodes - 1 - i;

			connection->_weight = distWeight(generator);

			connection->_innovationNumber = innovationNumber;

			innovationNumber++;

			_connectionSet.addConnection(minBias, maxBias, functionChances, connection, innovationNumber, generator);
		}
	}
	else if (dNodes < 0) {
		for (size_t i = 0; i < _connectionSet.getNumConnections();) {
			// If it was connected to an input that is now removed, delete the connection
			if (_connectionSet._connections[i]->_inIndex >= numOutputs || _connectionSet._connections[i]->_outIndex >= numOutputs) {
				// Remove references from nodes it connects
				NodeData &node = _connectionSet._nodes[_connectionSet._connections[i]->_inIndex];

				bool found = false;

				for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = node._connections.begin(); it != node._connections.end(); it++)
				if ((*it) == _connectionSet._connections[i]) {
					node._connections.erase(it);

					found = true;

					break;
				}

				//assert(found);

				found = false;

				node = _connectionSet._nodes[_connectionSet._connections[i]->_outIndex];

				for (std::list<std::shared_ptr<ConnectionGene>>::iterator it = node._connections.begin(); it != node._connections.end(); it++)
				if ((*it) == _connectionSet._connections[i]) {
					node._connections.erase(it);

					found = true;

					break;
				}

				//assert(found);

				_connectionSet._connections.erase(_connectionSet._connections.begin() + i);
			}
			else
				i++;
		}
	}

	_numOutputs = numOutputs;

	_connectionSet.setNumNodes(_numInputs + _numHidden + _numOutputs, minBias, maxBias, functionChances, generator);

#ifdef DEBUG
	for (size_t i = 0, numConnections = _connectionSet.GetNumConnections(); i < numConnections; i++)
		assert(_connectionSet._connections[i]->_outIndex >= _numInputs);
#endif

	updateNumHiddenNeurons();
}

void NetworkGenotype::connectUnconnectedInputs(float minWeight, float maxWeight, float minBias, float maxBias, const std::vector<float> &functionChances, InnovationNumberType &innovationNumber, std::mt19937 &generator) {
	// ---------------------------------- Add "Missing" Connections for Inputs and Outputs ----------------------------------

	const size_t numInputsHidden = _numInputs + _numHidden;

	std::uniform_real_distribution<float> distWeight(minWeight, maxWeight);

	// Inputs
	for (size_t i = 0; i < _numInputs; i++)
	// Input isn't connected to anything, add some connections
	if (_connectionSet._nodes[i]._connections.size() < _numOutputs)
	for (size_t j = 0; j < _numOutputs; j++) {
		std::shared_ptr<ConnectionGene> connection(new ConnectionGene());

		connection->_enabled = true;
		connection->_inIndex = i;
		connection->_outIndex = numInputsHidden + j;

		connection->_weight = distWeight(generator);

		connection->_innovationNumber = innovationNumber;

		innovationNumber++;

		_connectionSet.addConnection(minBias, maxBias, functionChances, connection, innovationNumber, generator);
	}
}


void NetworkGenotype::connectUnconnectedOutputs(float minWeight, float maxWeight, float minBias, float maxBias, const std::vector<float> &functionChances, InnovationNumberType &innovationNumber, std::mt19937 &generator) {
	// ---------------------------------- Add "Missing" Connections for Inputs and Outputs ----------------------------------

	updateNumHiddenNeurons();

	const size_t numInputsHidden = _numInputs + _numHidden;

	std::uniform_real_distribution<float> distWeight(minWeight, maxWeight);

	// Outputs
	for (size_t i = 0; i < _numOutputs; i++) {
		const size_t outputIndex = numInputsHidden + i;

		assert(outputIndex > 0 && outputIndex < _connectionSet._nodes.size());

		// Output isn't connected to anything, add some connections
		if (_connectionSet._nodes[outputIndex]._connections.size() < _numInputs)
		for (size_t j = 0; j < _numInputs; j++) {
			std::shared_ptr<ConnectionGene> connection(new ConnectionGene());

			connection->_enabled = true;
			connection->_inIndex = j;
			connection->_outIndex = outputIndex;

			connection->_weight = distWeight(generator);

			connection->_innovationNumber = innovationNumber;

			innovationNumber++;

			_connectionSet.addConnection(minBias, maxBias, functionChances, connection, innovationNumber, generator);
		}
	}
}

int NetworkGenotype::getNumUnconnectedInputs() const {
	int numUnconnected = 0;

	for (size_t i = 0; i < _numInputs; i++)
	if (_connectionSet._nodes[i]._connections.size() == 0)
		numUnconnected++;

	return numUnconnected;
}

int NetworkGenotype::getNumUnconnectedOutputs() const {
	int numUnconnected = 0;

	const size_t numInputsHidden = _numInputs + _numHidden;

	for (size_t i = 0; i < _numOutputs; i++)
	if (_connectionSet._nodes[numInputsHidden + i]._connections.size() == 0)
		numUnconnected++;

	return numUnconnected;
}

void NetworkGenotype::readFromStream(std::istream &is) {
	_connectionSet.removeConnections();
	_connectionSet._nodes.clear();

	// Read number of inputs and outputs
	is >> _numInputs >> _numOutputs >> _numHidden;

	// Read number of connections and number of nodes
	size_t numConnections, numNodes;
	is >> numConnections >> numNodes;

	_connectionSet._connections.reserve(numConnections);

	_connectionSet._nodes.resize(numNodes);

	// Read all connections
	for (size_t i = 0; i < numConnections; i++) {
		std::shared_ptr<ConnectionGene> connection(new ConnectionGene());

		is >> connection->_enabled >> connection->_inIndex >> connection->_outIndex >> connection->_weight >> connection->_innovationNumber;

		_connectionSet._connections.push_back(connection);

		// Add references to nodes
		assert(connection->_inIndex < numNodes);
		assert(connection->_outIndex < numNodes);

		_connectionSet._nodes[connection->_inIndex]._connections.push_back(connection);
		_connectionSet._nodes[connection->_outIndex]._connections.push_back(connection);
	}

	// Read all nodes (biases and innovation numbers)
	// These nodes will have already been added when the connections were loaded
	for (size_t i = 0; i < numNodes; i++)
		is >> _connectionSet._nodes[i]._bias >> _connectionSet._nodes[i]._innovationNumber;

	updateNumHiddenNeurons();
}

void NetworkGenotype::writeToStream(std::ostream &os) const {
	// Write number of inputs and outputs
	os << _numInputs << " " << _numOutputs << " " << _numHidden << std::endl;

	// Write number of connections and number of nodes
	os << _connectionSet.getNumConnections() << " " << _connectionSet.getNumNodes() << std::endl;

	// Write all connections
	for (size_t i = 0, numConnections = _connectionSet.getNumConnections(); i < numConnections; i++) {
		std::shared_ptr<ConnectionGene> connection = _connectionSet._connections[i];
		os << connection->_enabled << " " << connection->_inIndex << " " << connection->_outIndex << " " << connection->_weight << " " << connection->_innovationNumber << std::endl;
	}

	// Write all nodes (biases and innovation numbers)
	for (size_t i = 0, numNodes = _connectionSet.getNumNodes(); i < numNodes; i++) {
		const NodeData &node = _connectionSet._nodes[i];
		os << node._bias << " " << node._innovationNumber << std::endl;
	}
}