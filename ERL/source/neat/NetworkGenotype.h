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
*/

#pragma once

#include <neat/ConnectionGene.h>

#include <neat/Uncopyable.h>

#include <vector>
#include <list>

#include <random>
#include <memory>

// Uncomment the line below to allow disjoint/excess genes to become randomly enabled again
#define DISABLE_CHANCE_EXCESS_DISJOINT

// Uncomment line below to enable disjoint/excess gene similarity normalization
#define NORMALIZE_EXCESS_DISJOINT_SIMILARITY

namespace neat {
	class NetworkGenotype : public Uncopyable {
	public:
		struct NodeData {
			std::list<std::shared_ptr<ConnectionGene>> _connections;

			float _bias;

			InnovationNumberType _innovationNumber;

			NodeData()
				: _bias(0.0f), _innovationNumber(0)
			{}
		};

	private:
		size_t _numInputs, _numOutputs, _numHidden;

		class ConnectionSet {
		public:
			std::vector<std::shared_ptr<ConnectionGene>> _connections;

			std::vector<NodeData> _nodes;

			ConnectionSet();
			ConnectionSet(const ConnectionSet &other);

			ConnectionSet &operator=(const ConnectionSet &other);

			void addConnection(float minBias, float maxBias, const std::shared_ptr<ConnectionGene> &connection, InnovationNumberType &innovationNumber, std::mt19937 &generator);
			void addConnectionKnownBias(float bias1, float bias2, const std::shared_ptr<ConnectionGene> &connection, InnovationNumberType innovationNumber1, InnovationNumberType innovationNumber2);
			void removeConnections();

			void addNodes(int numNodes);
			void setNumNodes(size_t numNodes);

			size_t getNumNodes() const {
				return _nodes.size();
			}

			size_t getNumConnections() const {
				return _connections.size();
			}

			bool canSeverWithoutOrhpan(const ConnectionGene &connection) const;

		} _connectionSet;

		void removeUnusedNodes();

	public:
		bool _allowSelfConnections;

		NetworkGenotype();

		// For initializing starting genes
		void initialize(size_t numInputs, size_t numOutputs, float minWeight, float maxWeight, float minBias, float maxBias, InnovationNumberType &innovationNumber, std::mt19937 &generator); // Automatically increments innovation number

		void updateNumHiddenNeurons();

		void mutatePerturbWeight(float perturbationChance, float maxPerturbation, std::mt19937 &generator);
		void mutatePerturbWeightClamped(float perturbationChance, float maxPerturbation, float minWeight, float maxWeight, float minBias, float maxBias, std::mt19937 &generator);
		bool mutateAddConnection(float minWeight, float maxWeight, float minBias, float maxBias, InnovationNumberType &innovationNumber, std::mt19937 &generator); // Automatically increments innovation number. Returns false if cannot add connection
		void mutateAddNode(float minWeight, float maxWeight, float minBias, float maxBias, InnovationNumberType &innovationNumber, std::mt19937 &generator); // Automatically increments innovation number

		void crossover(const NetworkGenotype &otherParent, NetworkGenotype &child, float disableGeneChance, float fitnessForThis, float fitnessForOtherParent, std::mt19937 &generator); // Keeps parents, creates new child

		float getSimilarity(const NetworkGenotype &other, float excessFactor, float disjointFactor, float averageWeightDifferenceFactor, float inputCountDifferenceFactor, float outputCountDifferenceFactor);

		virtual void initializeAdditional(size_t numInputs, size_t numOutputs, float minWeight, float maxWeight, float minBias, float maxBias, InnovationNumberType &innovationNumber, std::mt19937 &generator) {}
		virtual void crossoverAdditional(const NetworkGenotype &otherParent, NetworkGenotype &child, float disableGeneChance, float fitnessForThis, float fitnessForOtherParent, std::mt19937 &generator) {}
		virtual void mutateAdditional(float perturbationChance, float maxPerturbation, float minWeight, float maxWeight, float minBias, float maxBias, InnovationNumberType &innovationNumber, std::mt19937 &generator) {}
		virtual float getSimilarityAdditional(const NetworkGenotype &other, float excessFactor, float disjointFactor, float averageWeightDifferenceFactor, float inputCountDifferenceFactor, float outputCountDifferenceFactor) { return 0.0f; }

		void setNumInputs(size_t numInputs);
		void setNumOutputs(size_t numOutputs);
		void setNumInputsFullyConnect(size_t numInputs, float minWeight, float maxWeight, float minBias, float maxBias, InnovationNumberType &innovationNumber, std::mt19937 &generator);
		void setNumOutputsFullyConnect(size_t numOutputs, float minWeight, float maxWeight, float minBias, float maxBias, InnovationNumberType &innovationNumber, std::mt19937 &generator);

		void connectUnconnectedInputs(float minWeight, float maxWeight, float minBias, float maxBias, InnovationNumberType &innovationNumber, std::mt19937 &generator);
		void connectUnconnectedOutputs(float minWeight, float maxWeight, float minBias, float maxBias, InnovationNumberType &innovationNumber, std::mt19937 &generator);

		int getNumUnconnectedInputs() const;
		int getNumUnconnectedOutputs() const;

		size_t getNumInputs() const {
			return _numInputs;
		}

		size_t getNumHidden() const {
			return _numHidden;
		}

		size_t getNumOutputs() const {
			return _numOutputs;
		}

		const std::vector<std::shared_ptr<ConnectionGene>> &getConnectionSet() const {
			return _connectionSet._connections;
		}

		const NodeData &getNodeData(size_t index) const {
			return _connectionSet._nodes[index];
		}

		void readFromStream(std::istream &is);
		void writeToStream(std::ostream &os) const;
	};
}
