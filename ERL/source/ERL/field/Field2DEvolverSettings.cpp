#include <erl/field/Field2DEvolverSettings.h>

using namespace erl;

Field2DEvolverSettings::Field2DEvolverSettings()
: _minInitWeight(-0.1f),
_maxInitWeight(0.1f),
_updateCrossoverAverageChance(0.1f),
_neAddNodeChance(0.1f),
_neAddConnectionChance(0.2f),
_neWeightPerturbationChance(0.15f),
_neWeightPerturbationStdDev(0.02f),
_neChangeFunctionChance(0.04f),
_neWeightFactor(0.5f),
_neDisjointFactor(1.0f),
_addConnectionResponseChance(0.04f),
_addNodeOutputChance(0.04f),
_addGasChance(0.05f),
_addTypeChance(0.05f),
_connectionReponseDifferenceFactor(5.0f),
_nodeOutputSizeDifferenceFactor(5.0f),
_gasCountDifferenceFactor(5.0f),
_typeSizeDifferenceFactor(5.0f),
_averageInitChance(0.1f),
_initPerturbationChance(0.05f),
_initPerturbationStdDev(0.025f),
_minInitInputStrengthScalar(10.0f),
_maxInitInputStrengthScalar(30.0f),
_averageInputStrengthScalarChance(0.3f),
_mutateInputStrengthChance(0.05f),
_inputStrengthPerturbationStdDev(1.0f),
_inputStrengthDifferenceFactor(1.0f),
_minInitConnectionStrengthScalar(2.0f),
_maxInitConnectionStrengthScalar(7.0f),
_averageConnectionStrengthScalarChance(0.2f),
_mutateConnectionStrengthChance(0.05f),
_connectionStrengthPerturbationStdDev(0.4f),
_connectionStrengthDifferenceFactor(1.0f),
_minInitNodeOutputStrengthScalar(2.0f),
_maxInitNodeOutputStrengthScalar(6.0f),
_averageNodeOutputStrengthScalarChance(0.3f),
_mutateNodeOutputStrengthChance(0.05f),
_nodeOutputStrengthPerturbationStdDev(0.4f),
_nodeOutputStrengthDifferenceFactor(1.0f)
{}

void Field2DEvolverSettings::readFromStream(std::istream &is) {
	std::string temp;

	is >> temp >> _minInitWeight;
	is >> temp >> _maxInitWeight;
	is >> temp >> _updateCrossoverAverageChance;
	is >> temp >> _neAddNodeChance;
	is >> temp >> _neAddConnectionChance;
	is >> temp >> _neWeightPerturbationChance;
	is >> temp >> _neWeightPerturbationStdDev;
	is >> temp >> _neChangeFunctionChance;
	is >> temp >> _neWeightFactor;
	is >> temp >> _neDisjointFactor;
	is >> temp >> _addConnectionResponseChance;
	is >> temp >> _addNodeOutputChance;
	is >> temp >> _addGasChance;
	is >> temp >> _addTypeChance;
	is >> temp >> _connectionReponseDifferenceFactor;
	is >> temp >> _nodeOutputSizeDifferenceFactor;
	is >> temp >> _gasCountDifferenceFactor;
	is >> temp >> _typeSizeDifferenceFactor;
	is >> temp >> _averageInitChance;
	is >> temp >> _initPerturbationChance;
	is >> temp >> _initPerturbationStdDev;
	is >> temp >> _minInitInputStrengthScalar;
	is >> temp >> _maxInitInputStrengthScalar;
	is >> temp >> _averageInputStrengthScalarChance;
	is >> temp >> _mutateInputStrengthChance;
	is >> temp >> _inputStrengthPerturbationStdDev;
	is >> temp >> _inputStrengthDifferenceFactor;
	is >> temp >> _minInitConnectionStrengthScalar;
	is >> temp >> _maxInitConnectionStrengthScalar;
	is >> temp >> _averageConnectionStrengthScalarChance;
	is >> temp >> _mutateConnectionStrengthChance;
	is >> temp >> _connectionStrengthPerturbationStdDev;
	is >> temp >> _connectionStrengthDifferenceFactor;
	is >> temp >> _minInitNodeOutputStrengthScalar;
	is >> temp >> _maxInitNodeOutputStrengthScalar;
	is >> temp >> _averageNodeOutputStrengthScalarChance;
	is >> temp >> _mutateNodeOutputStrengthChance;
	is >> temp >> _nodeOutputStrengthPerturbationStdDev;
	is >> temp >> _nodeOutputStrengthDifferenceFactor;
}