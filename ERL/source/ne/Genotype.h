#pragma once

#include <vector>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace ne {
	size_t roulette(const std::vector<float> &chances, std::mt19937 &generator);

	class Genotype {
	public:
		enum RemoveMethod {
			_random, _last
		};

		enum NodeType {
			_input, _output, _inputOutput, _hidden
		};

		struct NodePair {
			size_t _index0;
			size_t _index1;

			float _difference;
		};

		struct FunctionPair {
			size_t _functionIndex0;
			size_t _functionIndex1;

			bool operator==(const FunctionPair &other) const {
				return _functionIndex0 == other._functionIndex0 && _functionIndex1 == other._functionIndex1;
			}

			size_t operator()(const FunctionPair &p) const {
				return _functionIndex0 ^ _functionIndex1;
			}
		};

		struct Node {
			float _bias;
			size_t _functionIndex;

			std::unordered_map<size_t, float> _connections;
			std::unordered_set<size_t> _outputNodes;
		};

		static float getDifference(const Genotype &genotype0, const Genotype &genotype1, size_t node0ID, size_t node1ID, int searchDepth, float importanceDecay, float weightFactor, float disjointFactor, const std::unordered_map<FunctionPair, float, FunctionPair> &functionFactors, std::unordered_set<size_t> &visitedNodeIDs);

	private:
		std::unordered_map<size_t, std::shared_ptr<Node>> _nodes; // NodeID to node
		std::vector<size_t> _inputNodeIDs;
		std::vector<size_t> _outputNodeIDs;

		size_t _nextNodeID;

		void addNode(float minWeight, float maxWeight, const std::vector<float> &functionChances, std::mt19937 &generator);
		void addConnection(float minWeight, float maxWeight, std::mt19937 &generator);

	public:
		Genotype()
			: _nextNodeID(0)
		{}

		Genotype(const Genotype &other) {
			*this = other;
		}

		const Genotype &operator=(const Genotype &other);

		void createRandomFeedForward(size_t numInputs, size_t numOutputs, float minWeight, float maxWeight, const std::vector<float> &functionChances, std::mt19937 &generator);
		void createFromParents(const Genotype &parent0, const Genotype &parent1, float averageChance, std::mt19937 &generator);
		void mutate(float addNodeChance, float addConnectionChance, float minWeight, float maxWeight, float perturbationChance, float perturbationStdDev, float changeFunctionChance, const std::vector<float> &functionChances, std::mt19937 &generator);
	
		static float getDifference(const Genotype &genotype0, const Genotype &genotype1, float weightFactor, float disjointFactor, const std::unordered_map<FunctionPair, float, FunctionPair> &functionFactors);

		size_t getNumInputs() const {
			return _inputNodeIDs.size();
		}

		size_t getNumOutputs() const {
			return _outputNodeIDs.size();
		}

		void removeInput(size_t index);
		void removeOutput(size_t index);

		void addInputFeedForward(float minWeight, float maxWeight, const std::vector<float> &functionChances, std::mt19937 &generator);
		void addOutputFeedForward(float minWeight, float maxWeight, const std::vector<float> &functionChances, std::mt19937 &generator);

		void setNumInputsFeedForward(size_t numInputs, float minWeight, float maxWeight, const std::vector<float> &functionChances, std::mt19937 &generator, RemoveMethod removalMethod = _last);
		void setNumOutputsFeedForward(size_t numOutputs, float minWeight, float maxWeight, const std::vector<float> &functionChances, std::mt19937 &generator, RemoveMethod removalMethod = _last);

		void calculateOutgoingConnections();

		void readFromStream(std::istream &is);
		void writeToStream(std::ostream &os) const;

		friend class Phenotype;
	};
}