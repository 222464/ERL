#include <erl/field/Field2DGenes.h>

#include <erl/field/Field2DEvolverSettings.h>

#include<neat/Evolver.h>

#include <algorithm>

using namespace erl;

void Field2DGenes::initialize(size_t numInputs, size_t numOutputs, const neat::EvolverSettings* pSettings, const std::vector<float> &functionChances, neat::InnovationNumberType &innovationNumber, std::mt19937 &generator) {
	const Field2DEvolverSettings* pF2DSettings = static_cast<const Field2DEvolverSettings*>(pSettings);

	_connectionResponseSize = 1;
	_nodeOutputSize = 1;

	// + 3 is for type, random, and reward inputs
	_connectionUpdateGenotype.initialize(_nodeOutputSize + 3, _connectionResponseSize, pSettings, functionChances, innovationNumber, generator);
	_activationUpdateGenotype.initialize(_connectionResponseSize + 3, _nodeOutputSize, pSettings, functionChances, innovationNumber, generator);
	_typeSetGenotype.initialize(2, 1, pSettings, functionChances, innovationNumber, generator);
	_encoderGenotype.initialize(1, _connectionResponseSize, pSettings, functionChances, innovationNumber, generator);
	_decoderGenotype.initialize(_nodeOutputSize, 1, pSettings, functionChances, innovationNumber, generator);

	_recurrentNodeInitBounds.clear();
	_recurrentConnectionInitBounds.clear();
}

void Field2DGenes::crossover(const neat::EvolverSettings* pSettings, const std::vector<float> &functionChances, const Evolvable* pOtherParent, Evolvable* pChild, float fitnessForThis, float fitnessForOtherParent, neat::InnovationNumberType &innovationNumber, std::mt19937 &generator) {
	const Field2DEvolverSettings* pF2DSettings = static_cast<const Field2DEvolverSettings*>(pSettings);
	const Field2DGenes* pF2DOtherParent = static_cast<const Field2DGenes*>(pOtherParent);
	Field2DGenes* pF2DChild = static_cast<Field2DGenes*>(pChild);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	
	pF2DChild->_connectionResponseSize = dist01(generator) < 0.5f ? _connectionResponseSize : pF2DOtherParent->_connectionResponseSize;
	pF2DChild->_nodeOutputSize = dist01(generator) < 0.5f ? _nodeOutputSize : pF2DOtherParent->_nodeOutputSize;

	_connectionUpdateGenotype.crossover(pSettings, functionChances, &pF2DOtherParent->_connectionUpdateGenotype, &pF2DChild->_connectionUpdateGenotype, fitnessForThis, fitnessForOtherParent, innovationNumber, generator);
	_activationUpdateGenotype.crossover(pSettings, functionChances, &pF2DOtherParent->_activationUpdateGenotype, &pF2DChild->_activationUpdateGenotype, fitnessForThis, fitnessForOtherParent, innovationNumber, generator);
	_typeSetGenotype.crossover(pSettings, functionChances, &pF2DOtherParent->_typeSetGenotype, &pF2DChild->_typeSetGenotype, fitnessForThis, fitnessForOtherParent, innovationNumber, generator);
	_encoderGenotype.crossover(pSettings, functionChances, &pF2DOtherParent->_encoderGenotype, &pF2DChild->_encoderGenotype, fitnessForThis, fitnessForOtherParent, innovationNumber, generator);
	_decoderGenotype.crossover(pSettings, functionChances, &pF2DOtherParent->_decoderGenotype, &pF2DChild->_decoderGenotype, fitnessForThis, fitnessForOtherParent, innovationNumber, generator);

	pF2DChild->_connectionUpdateGenotype.setNumInputs(_nodeOutputSize + 3); // + 3 for type, random, and reward inputs
	pF2DChild->_connectionUpdateGenotype.setNumOutputs(_connectionResponseSize, pSettings->_minBias, pSettings->_maxBias, functionChances, generator);

	pF2DChild->_activationUpdateGenotype.setNumInputs(_connectionResponseSize + 3); // + 3 for type, random, and reward inputs
	pF2DChild->_activationUpdateGenotype.setNumOutputs(_nodeOutputSize, pSettings->_minBias, pSettings->_maxBias, functionChances, generator);

	pF2DChild->_encoderGenotype.setNumOutputs(_connectionResponseSize, pSettings->_minBias, pSettings->_maxBias, functionChances, generator);

	pF2DChild->_decoderGenotype.setNumInputs(_nodeOutputSize);

	// ------------------------ Initializations ------------------------

	if (_recurrentNodeInitBounds.size() > pF2DOtherParent->_recurrentNodeInitBounds.size())
		pF2DChild->_recurrentNodeInitBounds = _recurrentNodeInitBounds;
	else
		pF2DChild->_recurrentNodeInitBounds = pF2DOtherParent->_recurrentNodeInitBounds;

	int minRecurrentNodes = std::min(_recurrentNodeInitBounds.size(), pF2DOtherParent->_recurrentNodeInitBounds.size());

	for (int i = 0; i < minRecurrentNodes; i++) {
		// Crossover
		if (dist01(generator) < pF2DSettings->_averageInitChance) {
			std::get<0>(pF2DChild->_recurrentNodeInitBounds[i]) = (std::get<0>(_recurrentNodeInitBounds[i]) + std::get<0>(pF2DOtherParent->_recurrentNodeInitBounds[i])) * 0.5f;
			std::get<1>(pF2DChild->_recurrentNodeInitBounds[i]) = (std::get<1>(_recurrentNodeInitBounds[i]) + std::get<1>(pF2DOtherParent->_recurrentNodeInitBounds[i])) * 0.5f;
		}
		else
			pF2DChild->_recurrentNodeInitBounds[i] = dist01(generator) < 0.5f ? _recurrentNodeInitBounds[i] : pF2DOtherParent->_recurrentNodeInitBounds[i];
	}

	if (_recurrentConnectionInitBounds.size() > pF2DOtherParent->_recurrentConnectionInitBounds.size())
		pF2DChild->_recurrentConnectionInitBounds = _recurrentConnectionInitBounds;
	else
		pF2DChild->_recurrentConnectionInitBounds = pF2DOtherParent->_recurrentConnectionInitBounds;

	int minRecurrentConnections = std::min(_recurrentConnectionInitBounds.size(), pF2DOtherParent->_recurrentConnectionInitBounds.size());

	for (int i = 0; i < minRecurrentConnections; i++) {
		// Crossover
		if (dist01(generator) < pF2DSettings->_averageInitChance) {
			std::get<0>(pF2DChild->_recurrentConnectionInitBounds[i]) = (std::get<0>(_recurrentConnectionInitBounds[i]) + std::get<0>(pF2DOtherParent->_recurrentConnectionInitBounds[i])) * 0.5f;
			std::get<1>(pF2DChild->_recurrentConnectionInitBounds[i]) = (std::get<1>(_recurrentConnectionInitBounds[i]) + std::get<1>(pF2DOtherParent->_recurrentConnectionInitBounds[i])) * 0.5f;
		}
		else
			pF2DChild->_recurrentConnectionInitBounds[i] = dist01(generator) < 0.5f ? _recurrentConnectionInitBounds[i] : pF2DOtherParent->_recurrentConnectionInitBounds[i];
	}
}

void Field2DGenes::mutate(const neat::EvolverSettings* pSettings, const std::vector<float> &functionChances, neat::InnovationNumberType &innovationNumber, std::mt19937 &generator) {
	const Field2DEvolverSettings* pF2DSettings = static_cast<const Field2DEvolverSettings*>(pSettings);

	_connectionUpdateGenotype.mutate(pSettings, functionChances, innovationNumber, generator);
	_activationUpdateGenotype.mutate(pSettings, functionChances, innovationNumber, generator);
	_typeSetGenotype.mutate(pSettings, functionChances, innovationNumber, generator);
	_encoderGenotype.mutate(pSettings, functionChances, innovationNumber, generator);
	_decoderGenotype.mutate(pSettings, functionChances, innovationNumber, generator);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	if (dist01(generator) < pF2DSettings->_addConnectionResponseChance)
		_connectionResponseSize++;

	if (dist01(generator) < pF2DSettings->_addNodeOutputChance)
		_nodeOutputSize++;

	_connectionUpdateGenotype.setNumInputs(_nodeOutputSize + 3);
	_connectionUpdateGenotype.setNumOutputs(_connectionResponseSize, pSettings->_minBias, pSettings->_maxBias, functionChances, generator);

	_activationUpdateGenotype.setNumInputs(_connectionResponseSize + 3);
	_activationUpdateGenotype.setNumOutputs(_nodeOutputSize, pSettings->_minBias, pSettings->_maxBias, functionChances, generator);

	_encoderGenotype.setNumOutputs(_connectionResponseSize, pSettings->_minBias, pSettings->_maxBias, functionChances, generator);

	_decoderGenotype.setNumInputs(_nodeOutputSize);

	// ------------------------ Initializations ------------------------

	std::uniform_real_distribution<float> distPert(-pF2DSettings->_maxInitPerturbation, pF2DSettings->_maxInitPerturbation);

	for (int i = 0; i < _recurrentNodeInitBounds.size(); i++) {
		// Mutate
		std::get<0>(_recurrentNodeInitBounds[i]) += distPert(generator);
		std::get<1>(_recurrentNodeInitBounds[i]) += distPert(generator);

		if (std::get<0>(_recurrentNodeInitBounds[i]) > std::get<1>(_recurrentNodeInitBounds[i]))
			std::get<0>(_recurrentNodeInitBounds[i]) = std::get<1>(_recurrentNodeInitBounds[i]) = (std::get<0>(_recurrentNodeInitBounds[i]) + std::get<1>(_recurrentNodeInitBounds[i])) * 0.5f;
	}

	for (int i = 0; i < _recurrentConnectionInitBounds.size(); i++) {
		// Mutate
		std::get<0>(_recurrentConnectionInitBounds[i]) += distPert(generator);
		std::get<1>(_recurrentConnectionInitBounds[i]) += distPert(generator);

		if (std::get<0>(_recurrentConnectionInitBounds[i]) > std::get<1>(_recurrentConnectionInitBounds[i]))
			std::get<0>(_recurrentConnectionInitBounds[i]) = std::get<1>(_recurrentConnectionInitBounds[i]) = (std::get<0>(_recurrentConnectionInitBounds[i]) + std::get<1>(_recurrentConnectionInitBounds[i])) * 0.5f;
	}
}

float Field2DGenes::getSimilarity(const neat::EvolverSettings* pSettings, const std::vector<float> &functionChances, const Evolvable* pOther) {
	const Field2DEvolverSettings* pF2DSettings = static_cast<const Field2DEvolverSettings*>(pSettings);
	const Field2DGenes* pF2DOther = static_cast<const Field2DGenes*>(pOther);

	return std::abs(_connectionResponseSize - pF2DOther->_connectionResponseSize) * pF2DSettings->_connectionReponseDifferenceFactor +
		std::abs(_nodeOutputSize - pF2DOther->_nodeOutputSize) * pF2DSettings->_nodeOutputSizeDifferenceFactor +
		_connectionUpdateGenotype.getSimilarity(pSettings, functionChances, &pF2DOther->_connectionUpdateGenotype) +
		_activationUpdateGenotype.getSimilarity(pSettings, functionChances, &pF2DOther->_activationUpdateGenotype) +
		_typeSetGenotype.getSimilarity(pSettings, functionChances, &pF2DOther->_typeSetGenotype) +
		_encoderGenotype.getSimilarity(pSettings, functionChances, &pF2DOther->_encoderGenotype) +
		_decoderGenotype.getSimilarity(pSettings, functionChances, &pF2DOther->_decoderGenotype);
}