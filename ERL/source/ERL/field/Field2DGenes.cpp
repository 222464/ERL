#include <erl/field/Field2DGenes.h>

#include <algorithm>

using namespace erl;

void Field2DGenes::setInputOutputCounts(const Field2DEvolverSettings* pSettings, const std::vector<float> &functionChances, std::mt19937 &generator) {
	_connectionUpdateGenotype.setNumInputsFeedForward(_nodeOutputSize + 6, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);
	_connectionUpdateGenotype.setNumOutputsFeedForward(_connectionResponseSize, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);

	_activationUpdateGenotype.setNumInputsFeedForward(_connectionResponseSize + 3 + _numGases, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);
	_activationUpdateGenotype.setNumOutputsFeedForward(_nodeOutputSize + _numGases, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);

	_typeSetGenotype.setNumInputsFeedForward(2, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);
	_typeSetGenotype.setNumOutputsFeedForward(1, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);

	_encoderGenotype.setNumInputsFeedForward(1, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);
	_encoderGenotype.setNumOutputsFeedForward(_connectionResponseSize, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);

	_decoderGenotype.setNumInputsFeedForward(_nodeOutputSize, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);
	_decoderGenotype.setNumOutputsFeedForward(1, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);
}

void Field2DGenes::initialize(const Field2DEvolverSettings* pSettings, const std::vector<float> &functionChances, std::mt19937 &generator) {
	const Field2DEvolverSettings* pF2DSettings = static_cast<const Field2DEvolverSettings*>(pSettings);

	_connectionResponseSize = 1;
	_nodeOutputSize = 1;
	_numGases = 1;

	std::uniform_real_distribution<float> inputStrengthScalarDist(pF2DSettings->_minInitInputStrengthScalar, pF2DSettings->_maxInitInputStrengthScalar);
	std::uniform_real_distribution<float> connectionStrengthScalarDist(pF2DSettings->_minInitConnectionStrengthScalar, pF2DSettings->_maxInitConnectionStrengthScalar);
	std::uniform_real_distribution<float> nodeOutputStrengthScalarDist(pF2DSettings->_minInitNodeOutputStrengthScalar, pF2DSettings->_maxInitNodeOutputStrengthScalar);

	_inputStrengthScalar = inputStrengthScalarDist(generator);
	_connectionStrengthScalar = connectionStrengthScalarDist(generator);
	_nodeOutputStrengthScalar = nodeOutputStrengthScalarDist(generator);

	// + 3 is for type, random, and reward inputs. + 6 is 1 additional type as well as delta position for connections
	_connectionUpdateGenotype.createRandomFeedForward(_nodeOutputSize + 6, _connectionResponseSize, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);
	_activationUpdateGenotype.createRandomFeedForward(_connectionResponseSize + 3 + _numGases, _nodeOutputSize + _numGases, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);
	_typeSetGenotype.createRandomFeedForward(2, 1, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);
	_encoderGenotype.createRandomFeedForward(1, _connectionResponseSize, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);
	_decoderGenotype.createRandomFeedForward(_nodeOutputSize, 1, pSettings->_minInitWeight, pSettings->_maxInitWeight, functionChances, generator);

	_recurrentNodeInitBounds.clear();
	_recurrentConnectionInitBounds.clear();
}

void Field2DGenes::crossover(const Field2DEvolverSettings* pSettings, const std::vector<float> &functionChances, const Field2DGenes* pParent1, const Field2DGenes* pParent2, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	
	_connectionResponseSize = dist01(generator) < 0.5f ? pParent1->_connectionResponseSize : pParent2->_connectionResponseSize;
	_nodeOutputSize = dist01(generator) < 0.5f ? pParent1->_nodeOutputSize : pParent2->_nodeOutputSize;
	_numGases = dist01(generator) < 0.5f ? pParent1->_numGases : pParent2->_numGases;

	_inputStrengthScalar = dist01(generator) < pSettings->_averageInputStrengthScalarChance ? (pParent1->_inputStrengthScalar + pParent2->_inputStrengthScalar) * 0.5f : (dist01(generator) < 0.5f ? pParent1->_inputStrengthScalar : pParent2->_inputStrengthScalar);
	_connectionStrengthScalar = dist01(generator) < pSettings->_averageConnectionStrengthScalarChance ? (pParent1->_connectionStrengthScalar + pParent2->_connectionStrengthScalar) * 0.5f : (dist01(generator) < 0.5f ? pParent1->_connectionStrengthScalar : pParent2->_connectionStrengthScalar);
	_nodeOutputStrengthScalar = dist01(generator) < pSettings->_averageNodeOutputStrengthScalarChance ? (pParent1->_nodeOutputStrengthScalar + pParent2->_nodeOutputStrengthScalar) * 0.5f : (dist01(generator) < 0.5f ? pParent1->_nodeOutputStrengthScalar : pParent2->_nodeOutputStrengthScalar);

	_connectionUpdateGenotype.createFromParents(pParent1->_connectionUpdateGenotype, pParent2->_connectionUpdateGenotype, pSettings->_updateCrossoverAverageChance, generator);
	_activationUpdateGenotype.createFromParents(pParent1->_activationUpdateGenotype, pParent2->_activationUpdateGenotype, pSettings->_updateCrossoverAverageChance, generator);
	_typeSetGenotype.createFromParents(pParent1->_typeSetGenotype, pParent2->_typeSetGenotype, pSettings->_updateCrossoverAverageChance, generator);
	_encoderGenotype.createFromParents(pParent1->_encoderGenotype, pParent2->_encoderGenotype, pSettings->_updateCrossoverAverageChance, generator);
	_decoderGenotype.createFromParents(pParent1->_decoderGenotype, pParent2->_decoderGenotype, pSettings->_updateCrossoverAverageChance, generator);

	setInputOutputCounts(pSettings, functionChances, generator);

	// ------------------------ Initializations ------------------------

	if (pParent1->_recurrentNodeInitBounds.size() > pParent2->_recurrentNodeInitBounds.size())
		_recurrentNodeInitBounds = pParent1->_recurrentNodeInitBounds;
	else
		_recurrentNodeInitBounds = pParent2->_recurrentNodeInitBounds;

	int minRecurrentNodes = std::min(pParent1->_recurrentNodeInitBounds.size(), pParent2->_recurrentNodeInitBounds.size());

	for (int i = 0; i < minRecurrentNodes; i++) {
		// Crossover
		if (dist01(generator) < pSettings->_averageInitChance) {
			std::get<0>(_recurrentNodeInitBounds[i]) = (std::get<0>(pParent1->_recurrentNodeInitBounds[i]) + std::get<0>(pParent2->_recurrentNodeInitBounds[i])) * 0.5f;
			std::get<1>(_recurrentNodeInitBounds[i]) = (std::get<1>(pParent1->_recurrentNodeInitBounds[i]) + std::get<1>(pParent2->_recurrentNodeInitBounds[i])) * 0.5f;
		}
		else
			_recurrentNodeInitBounds[i] = dist01(generator) < 0.5f ? pParent1->_recurrentNodeInitBounds[i] : pParent2->_recurrentNodeInitBounds[i];
	}

	if (pParent1->_recurrentConnectionInitBounds.size() > pParent2->_recurrentConnectionInitBounds.size())
		_recurrentConnectionInitBounds = pParent1->_recurrentConnectionInitBounds;
	else
		_recurrentConnectionInitBounds = pParent2->_recurrentConnectionInitBounds;

	int minRecurrentConnections = std::min(pParent1->_recurrentConnectionInitBounds.size(), pParent2->_recurrentConnectionInitBounds.size());

	for (int i = 0; i < minRecurrentConnections; i++) {
		// Crossover
		if (dist01(generator) < pSettings->_averageInitChance) {
			std::get<0>(_recurrentConnectionInitBounds[i]) = (std::get<0>(pParent1->_recurrentConnectionInitBounds[i]) + std::get<0>(pParent2->_recurrentConnectionInitBounds[i])) * 0.5f;
			std::get<1>(_recurrentConnectionInitBounds[i]) = (std::get<1>(pParent1->_recurrentConnectionInitBounds[i]) + std::get<1>(pParent2->_recurrentConnectionInitBounds[i])) * 0.5f;
		}
		else
			_recurrentConnectionInitBounds[i] = dist01(generator) < 0.5f ? pParent1->_recurrentConnectionInitBounds[i] : pParent2->_recurrentConnectionInitBounds[i];
	}
}

void Field2DGenes::mutate(const Field2DEvolverSettings* pSettings, const std::vector<float> &functionChances, std::mt19937 &generator) {
	const Field2DEvolverSettings* pF2DSettings = static_cast<const Field2DEvolverSettings*>(pSettings);

	_connectionUpdateGenotype.mutate(pSettings->_neAddNodeChance, pSettings->_neAddConnectionChance, pSettings->_minInitWeight, pSettings->_maxInitWeight, pSettings->_neWeightPerturbationChance, pSettings->_neMaxWeightPerturbation, pSettings->_neChangeFunctionChance, functionChances, generator);
	_activationUpdateGenotype.mutate(pSettings->_neAddNodeChance, pSettings->_neAddConnectionChance, pSettings->_minInitWeight, pSettings->_maxInitWeight, pSettings->_neWeightPerturbationChance, pSettings->_neMaxWeightPerturbation, pSettings->_neChangeFunctionChance, functionChances, generator);
	_typeSetGenotype.mutate(pSettings->_neAddNodeChance, pSettings->_neAddConnectionChance, pSettings->_minInitWeight, pSettings->_maxInitWeight, pSettings->_neWeightPerturbationChance, pSettings->_neMaxWeightPerturbation, pSettings->_neChangeFunctionChance, functionChances, generator);
	_encoderGenotype.mutate(pSettings->_neAddNodeChance, pSettings->_neAddConnectionChance, pSettings->_minInitWeight, pSettings->_maxInitWeight, pSettings->_neWeightPerturbationChance, pSettings->_neMaxWeightPerturbation, pSettings->_neChangeFunctionChance, functionChances, generator);
	_decoderGenotype.mutate(pSettings->_neAddNodeChance, pSettings->_neAddConnectionChance, pSettings->_minInitWeight, pSettings->_maxInitWeight, pSettings->_neWeightPerturbationChance, pSettings->_neMaxWeightPerturbation, pSettings->_neChangeFunctionChance, functionChances, generator);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	if (dist01(generator) < pF2DSettings->_addConnectionResponseChance)
		_connectionResponseSize++;

	if (dist01(generator) < pF2DSettings->_addNodeOutputChance)
		_nodeOutputSize++;

	if (dist01(generator) < pF2DSettings->_addGasChance)
		_numGases++;

	if (dist01(generator) < pF2DSettings->_mutateInputStrengthChance) {
		std::uniform_real_distribution<float> inputStrengthPertDist(-pF2DSettings->_maxInputStrengthPerturbation, pF2DSettings->_maxInputStrengthPerturbation);

		_inputStrengthScalar += inputStrengthPertDist(generator);
	}

	if (dist01(generator) < pF2DSettings->_mutateConnectionStrengthChance) {
		std::uniform_real_distribution<float> connectionStrengthPertDist(-pF2DSettings->_maxConnectionStrengthPerturbation, pF2DSettings->_maxConnectionStrengthPerturbation);

		_connectionStrengthScalar += connectionStrengthPertDist(generator);
	}

	if (dist01(generator) < pF2DSettings->_mutateNodeOutputStrengthChance) {
		std::uniform_real_distribution<float> nodeOutputStrengthPertDist(-pF2DSettings->_maxNodeOutputStrengthPerturbation, pF2DSettings->_maxNodeOutputStrengthPerturbation);

		_nodeOutputStrengthScalar += nodeOutputStrengthPertDist(generator);
	}

	setInputOutputCounts(pSettings, functionChances, generator);

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

float Field2DGenes::getSimilarity(const Field2DEvolverSettings* pSettings, const std::vector<float> &functionChances, const Field2DGenes* pGenotype1, const Field2DGenes* pGenotype2, const std::unordered_map<ne::Genotype::FunctionPair, float, ne::Genotype::FunctionPair> &functionFactors) {
	const Field2DEvolverSettings* pF2DSettings = static_cast<const Field2DEvolverSettings*>(pSettings);

	return std::abs(pGenotype1->_connectionResponseSize - pGenotype2->_connectionResponseSize) * pF2DSettings->_connectionReponseDifferenceFactor +
		std::abs(pGenotype1->_nodeOutputSize - pGenotype2->_nodeOutputSize) * pF2DSettings->_nodeOutputSizeDifferenceFactor +
		std::abs(pGenotype1->_numGases - pGenotype2->_numGases) * pF2DSettings->_gasCountDifferenceFactor +
		std::abs(pGenotype1->_inputStrengthScalar - pGenotype2->_inputStrengthScalar) * pF2DSettings->_inputStrengthDifferenceFactor +
		std::abs(pGenotype1->_connectionStrengthScalar - pGenotype2->_connectionStrengthScalar) * pF2DSettings->_connectionStrengthDifferenceFactor +
		std::abs(pGenotype1->_nodeOutputStrengthScalar - pGenotype2->_nodeOutputStrengthScalar) * pF2DSettings->_nodeOutputStrengthDifferenceFactor +
		ne::Genotype::getDifference(pGenotype1->_connectionUpdateGenotype, pGenotype2->_connectionUpdateGenotype, pSettings->_neWeightFactor, pSettings->_neDisjointFactor, functionFactors) +
		ne::Genotype::getDifference(pGenotype1->_activationUpdateGenotype, pGenotype2->_activationUpdateGenotype, pSettings->_neWeightFactor, pSettings->_neDisjointFactor, functionFactors) +
		ne::Genotype::getDifference(pGenotype1->_typeSetGenotype, pGenotype2->_typeSetGenotype, pSettings->_neWeightFactor, pSettings->_neDisjointFactor, functionFactors) +
		ne::Genotype::getDifference(pGenotype1->_encoderGenotype, pGenotype2->_encoderGenotype, pSettings->_neWeightFactor, pSettings->_neDisjointFactor, functionFactors) +
		ne::Genotype::getDifference(pGenotype1->_decoderGenotype, pGenotype2->_decoderGenotype, pSettings->_neWeightFactor, pSettings->_neDisjointFactor, functionFactors);
}

void Field2DGenes::readFromStream(std::istream &is) {
	is >> _connectionResponseSize >> _nodeOutputSize >> _numGases;
	is >> _inputStrengthScalar >> _connectionStrengthScalar >> _nodeOutputStrengthScalar;

	int numRecurrentNode;
	int numRecurrentConnection;

	is >> numRecurrentNode >> numRecurrentConnection;

	_recurrentNodeInitBounds.resize(numRecurrentNode);
	_recurrentConnectionInitBounds.resize(numRecurrentConnection);

	for (int i = 0; i < numRecurrentNode; i++) {
		float v0, v1;
		is >> v0 >> v1;
		_recurrentNodeInitBounds[i] = std::make_tuple(v0, v1);
	}

	for (int i = 0; i < numRecurrentConnection; i++) {
		float v0, v1;
		is >> v0 >> v1;
		_recurrentConnectionInitBounds[i] = std::make_tuple(v0, v1);
	}

	_connectionUpdateGenotype.readFromStream(is);
	_activationUpdateGenotype.readFromStream(is);
	_typeSetGenotype.readFromStream(is);
	_encoderGenotype.readFromStream(is);
	_decoderGenotype.readFromStream(is);
}

void Field2DGenes::writeToStream(std::ostream &os) const {
	os << _connectionResponseSize << " " << _nodeOutputSize << " " << _numGases << std::endl;
	os << _inputStrengthScalar << " " << _connectionStrengthScalar << " " << _nodeOutputStrengthScalar << std::endl;

	os << _recurrentNodeInitBounds.size() << " " << _recurrentConnectionInitBounds.size() << std::endl;

	for (int i = 0; i < _recurrentNodeInitBounds.size(); i++) {
		os << std::get<0>(_recurrentNodeInitBounds[i]) << " " << std::get<1>(_recurrentNodeInitBounds[i]) << " ";
	}

	os << std::endl;

	for (int i = 0; i < _recurrentConnectionInitBounds.size(); i++) {
		os << std::get<0>(_recurrentConnectionInitBounds[i]) << " " << std::get<1>(_recurrentConnectionInitBounds[i]) << " ";
	}

	os << std::endl << std::endl;;

	_connectionUpdateGenotype.writeToStream(os);
	os << std::endl;
	_activationUpdateGenotype.writeToStream(os);
	os << std::endl;
	_typeSetGenotype.writeToStream(os);
	os << std::endl;
	_encoderGenotype.writeToStream(os);
	os << std::endl;
	_decoderGenotype.writeToStream(os);
}