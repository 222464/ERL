#include <erl/field/Field2DEvolverSettings.h>

using namespace erl;

Field2DEvolverSettings::Field2DEvolverSettings()
: _addConnectionResponseChance(0.02f),
_addNodeOutputChance(0.02f),
_addGasChance(0.02f),
_connectionReponseDifferenceFactor(5.0f),
_nodeOutputSizeDifferenceFactor(5.0f),
_gasCountDifferenceFactor(5.0f),
_averageInitChance(0.2f),
_maxInitPerturbation(0.05f),
_minInitInputStrengthScalar(8.0f),
_maxInitInputStrengthScalar(20.0f),
_averageInputStrengthScalarChance(0.2f),
_mutateInputStrengthChance(0.2f),
_maxInputStrengthPerturbation(4.0f)
{}