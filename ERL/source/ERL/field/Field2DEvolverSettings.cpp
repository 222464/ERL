#include <erl/field/Field2DEvolverSettings.h>

using namespace erl;

Field2DEvolverSettings::Field2DEvolverSettings()
: _addConnectionResponseChance(0.02f),
_addNodeOutputChance(0.02f),
_connectionReponseDifferenceFactor(5.0f),
_nodeOutputSizeDifferenceFactor(5.0f),
_averageInitChance(0.3f),
_maxInitPerturbation(0.5f)
{}