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
		float _inputStrengthDifferenceFactor;

		float _minInitConnectionStrengthScalar;
		float _maxInitConnectionStrengthScalar;
		float _averageConnectionStrengthScalarChance;
		float _mutateConnectionStrengthChance;
		float _maxConnectionStrengthPerturbation;
		float _connectionStrengthDifferenceFactor;

		float _minInitNodeOutputStrengthScalar;
		float _maxInitNodeOutputStrengthScalar;
		float _averageNodeOutputStrengthScalarChance;
		float _mutateNodeOutputStrengthChance;
		float _maxNodeOutputStrengthPerturbation;
		float _nodeOutputStrengthDifferenceFactor;

		Field2DEvolverSettings();
	};
}
