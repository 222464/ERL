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

#include <neat/Neuron.h>
#include <neat/NetworkGenotype.h>

#include <vector>
#include <memory>

namespace neat {
	class NetworkPhenotype {
	private:
		std::vector<NeuronInput> _inputs;
		std::vector<Neuron> _hidden; // Can be recurrent with themselves and output layer
		std::vector<Neuron> _outputs;

	public:
		float _activationMultiplier;

		NetworkPhenotype();

		// Treat inputs, hidden, and outputs as one array with these helpers.
		// Need two though, since the types vary based on whether it is an input or not
		NeuronInput &getNeuronInputNode(size_t index);
		Neuron &getNeuronNode(size_t index);

		void create(size_t numInputs, size_t numOutputs, const std::vector<std::shared_ptr<ConnectionGene>> &connections, size_t numHidden);
		void create(const NetworkGenotype &genotype);

		void update();

		size_t NetworkPhenotype::getNumInputs() const {
			return _inputs.size();
		}

		size_t NetworkPhenotype::getNumHidden() const {
			return _hidden.size();
		}

		size_t NetworkPhenotype::getNumOutputs() const {
			return _outputs.size();
		}

		NeuronInput &NetworkPhenotype::getInput(size_t inputIndex) {
			return _inputs[inputIndex];
		}

		const Neuron &NetworkPhenotype::getOutput(size_t outputIndex) const {
			return _outputs[outputIndex];
		}

		void resetOutputs();
	};
}