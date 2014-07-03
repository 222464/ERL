/*
ERL

Field2D
*/

#pragma once

#include <erl/platform/ComputeSystem.h>
#include <erl/field/Field2DGenes.h>
#include <erl/platform/SoftwareImage2D.h>
#include <neat/NetworkPhenotype.h>
#include <array>

namespace erl {
	class Field2D {
	public:
		struct RandomSeed {
			float _x, _y;

			RandomSeed() {}
			RandomSeed(float x, float y)
				: _x(x), _y(y)
			{}
		};

	private:
		std::array<cl::Buffer, 2> _buffers;
		cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Image2D&, cl::Image1D&, cl::Image1D&, cl::Image2D&, RandomSeed, float> _kernelFunctor;

		unsigned char _currentReadBufferIndex;
		unsigned char _currentWriteBufferIndex;

		cl::Program _program;

		cl::Image2D _typeImage;

		cl::Image1D _inputImage;
		cl::Image1D _outputImage;

		std::shared_ptr<cl::Image2D> _randomImage;

		neat::NetworkPhenotype _connectionPhenotype;
		neat::NetworkPhenotype _nodePhenotype;

		neat::NetworkPhenotype::RuleData _connectionData;
		neat::NetworkPhenotype::RuleData _nodeData;

		int _connectionResponseSize;
		int _nodeOutputSize;
		int _width, _height;
		int _connectionRadius;

		int _connectionDimensionSize;
		int _numConnections;

		int _nodeSize;
		int _connectionSize;
		int _nodeAndConnectionsSize;

		int _numNodes;

		int _bufferSize;

		std::vector<float> _inputs;
		std::vector<float> _outputs;

		std::vector<neat::NetworkPhenotype> _encoderPhenotypes;
		std::vector<neat::NetworkPhenotype> _decoderPhenotypes;

	public:
		void create(Field2DGenes &genes, ComputeSystem &cs, int width, int height, int connectionRadius, int numInputs, int numOutputs, const std::shared_ptr<cl::Image2D> &randomImage,
			const std::vector<std::function<float(float)>> &activationFunctions, float minRecInit, float maxRecInit, float inputRadius, std::mt19937 &generator);

		void update(float reward, ComputeSystem &cs, const std::vector<std::function<float(float)>> &activationFunctions, std::mt19937 &generator);

		const neat::NetworkPhenotype::RuleData &getConnectionData() const {
			return _connectionData;
		}

		const neat::NetworkPhenotype::RuleData &getNodeData() const {
			return _nodeData;
		}

		int getWidth() const {
			return _width;
		}

		int getHeight() const {
			return _height;
		}

		int getConnectionRadius() const {
			return _connectionRadius;
		}

		int getConnectionDimensionSize() const {
			return _connectionDimensionSize;
		}

		int getNumConnections() const {
			return _numConnections;
		}

		int getNodeSize() const {
			return _nodeSize;
		}

		int getConnectionSize() const {
			return _connectionSize;
		}

		int getNodeAndConnectionsSize() const {
			return _nodeAndConnectionsSize;
		}

		int getNumNodes() const {
			return _numNodes;
		}

		int getBufferSize() const {
			return _bufferSize;
		}

		int getNumInputs() const {
			return _inputs.size();
		}

		int getNumOutputs() const {
			return _outputs.size();
		}

		void setInput(int index, float value) {
			_inputs[index] = value;
		}

		float getOutput(int index) const {
			return _outputs[index];
		}
	};
}