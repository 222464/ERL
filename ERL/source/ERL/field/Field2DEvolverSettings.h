/*
ERL

Field2DEvolverSettings
*/

#pragma once

namespace erl {
	class Field2DEvolverSettings {
	public:
		float _minInitWeight;
		float _maxInitWeight;
		float _updateCrossoverAverageChance;
		float _neAddNodeChance;
		float _neAddConnectionChance;
		float _neWeightPerturbationChance;
		float _neMaxWeightPerturbation;
		float _neChangeFunctionChance;

		float _neWeightFactor;
		float _neDisjointFactor;

		// For expanding node output and connection output sizes over time
		float _addConnectionResponseChance;
		float _addNodeOutputChance;
		float _addGasChance;

		float _connectionReponseDifferenceFactor;
		float _nodeOutputSizeDifferenceFactor;
		float _gasCountDifferenceFactor;

		float _averageInitChance;
		float _initPerturbationChance;
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
