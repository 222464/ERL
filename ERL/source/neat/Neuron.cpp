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

#include <neat/Neuron.h>

#include <neat/NetworkPhenotype.h>

#include <iostream>

using namespace neat;

Neuron::Neuron()
: _bias(0.0f)
{}

void Neuron::update(NetworkPhenotype &phenotype) {
	float sum = _bias;
	
	for (size_t i = 0; i < _inputs.size(); i++)
		sum += phenotype.getNeuronInputNode(_inputs[i]._inputOffset)._output * _inputs[i]._weight;

	_output = sigmoid(sum * phenotype._activationMultiplier);
}

std::ostream &operator<<(std::ostream &os, Neuron &neuron) {
	os << "W: ";

	for (size_t i = 0; i < neuron._inputs.size(); i++)
		os << neuron._inputs[i]._weight << ", ";

	os << "B: " << neuron._bias;

	return os;
}