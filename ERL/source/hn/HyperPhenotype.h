#pragma once

#include <ne/Phenotype.h>

#include <assert.h>

namespace hn {
	int getIndex(const std::vector<int> &dimensions, const std::vector<int> &coordinate);
	void getCoordinate(const std::vector<int> &dimensions, int index, std::vector<int> &coordinate);

	class HyperPhenotype {
	public:
		enum NodeType {
			_input = 0, _neuron = 1
		};

		struct Connection {
			int _index;
			float _weight;
		};

		struct Node {
			std::vector<Connection> _connections;
			float _bias;

			float _state;
			float _statePrev;

			NodeType _type;

			Node()
				: _state(0.0f), _statePrev(0.0f), _type(_neuron)
			{}
		};

		float sigmoid(float x) {
			return 1.0f / (1.0f + std::exp(-x));
		}

	private:
		std::vector<int> _substrateDimensions;
		std::vector<int> _inputIndices;

		std::vector<Node> _nodes;

	public:
		void createFromGenotype(const ne::Genotype &connectionGenotype, const ne::Genotype &biasGenotype, const std::vector<int> &substrateDimensions, const std::vector<int> &inputIndices, const std::vector<std::function<float(float)>> &functions, int connectionRadius, float weightThreshold);

		void setInput(int index, float value) {
			assert(_nodes[index]._type == _input);
			_nodes[index]._state = value;
		}

		float getState(int index) const {
			return _nodes[index]._state;
		}

		void update();

		void clearStates();
	};
}