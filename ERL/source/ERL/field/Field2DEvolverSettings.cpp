#include <erl/field/Field2DEvolverSettings.h>

using namespace erl;

Field2DEvolverSettings::Field2DEvolverSettings()
: _addConnectionResponseChance(0.04f),
_addNodeOutputChance(0.04f),
_addGasChance(0.02f),
_connectionReponseDifferenceFactor(5.0f),
_nodeOutputSizeDifferenceFactor(5.0f),
_gasCountDifferenceFactor(5.0f),
_averageInitChance(0.2f),
_maxInitPerturbation(0.05f),
_minInitInputStrengthScalar(10.0f),
_maxInitInputStrengthScalar(20.0f),
_averageInputStrengthScalarChance(0.1f),
_mutateInputStrengthChance(0.1f),
_maxInputStrengthPerturbation(4.0f),
_inputStrengthDifferenceFactor(1.0f),
_minInitConnectionStrengthScalar(2.0f),
_maxInitConnectionStrengthScalar(5.0f),
_averageConnectionStrengthScalarChance(0.2f),
_mutateConnectionStrengthChance(0.1f),
_maxConnectionStrengthPerturbation(1.0f),
_connectionStrengthDifferenceFactor(1.0f),
_minInitNodeOutputStrengthScalar(2.0f),
_maxInitNodeOutputStrengthScalar(6.0f),
_averageNodeOutputStrengthScalarChance(0.2f),
_mutateNodeOutputStrengthChance(0.1f),
_maxNodeOutputStrengthPerturbation(1.0f),
_nodeOutputStrengthDifferenceFactor(1.0f)
{}