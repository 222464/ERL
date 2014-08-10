#pragma once

#include <ne/Genotype.h>

#include <functional>

namespace ne {
	class Phenotype {
	public:
		enum FetchType {
			_input, _recurrent, _intermediate
		};

		struct Connection {
			FetchType _fetchType;
			size_t _fetchIndex;
			float _weight;
		};

		struct Node {
			std::vector<Connection> _connections;
			size_t _functionIndex;
			float _bias;

			float _output;
		};

	private:
		size_t _numInputs, _numOutputs;

		std::vector<std::shared_ptr<Node>> _nodes;

		std::vector<size_t> _recurrentNodeIndices;

	public:
		Phenotype();

		void createFromGenotype(const Genotype &genotype);

		void execute(const std::vector<float> &inputs, std::vector<float> &outputs, std::vector<float> &recurrentData, const std::vector<std::function<float(float)>> &functions);

		size_t getNumInputs() const {
			return _numInputs;
		}

		size_t getNumOutputs() const {
			return _numOutputs;
		}

		size_t getRecurrentDataSize() const {
			return _recurrentNodeIndices.size();
		}

		const std::vector<std::shared_ptr<Node>> &getNodes() const {
			return _nodes;
		}

		const std::vector<size_t> &getRecurrentNodeIndices() const {
			return _recurrentNodeIndices;
		}
	};
}