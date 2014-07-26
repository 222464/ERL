/*
ERL

Field2DEvolverSettings
*/

#include <neat/Evolver.h>

namespace erl {
	class Field2DEvolverSettings : public neat::EvolverSettings {
	public:
		// For expanding node output and connection output sizes over time
		float _addConnectionResponseChance;
		float _addNodeOutputChance;
		float _addGasChance;

		float _connectionReponseDifferenceFactor;
		float _nodeOutputSizeDifferenceFactor;
		float _gasCountDifferenceFactor;

		float _averageInitChance;
		float _maxInitPerturbation;

		float _minInitInputStrengthScalar;
		float _maxInitInputStrengthScalar;
		float _averageInputStrengthScalarChance;
		float _mutateInputStrengthChance;
		float _maxInputStrengthPerturbation;

		Field2DEvolverSettings();
	};
}
