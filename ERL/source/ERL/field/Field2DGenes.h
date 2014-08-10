/*
ERL

Field2D Genes
*/

#pragma once

#include <erl/field/Field2DEvolverSettings.h>
#include <ne/Genotype.h>

namespace erl {
	class Field2DGenes {
	private:
		ne::Genotype _connectionUpdateGenotype;
		ne::Genotype _activationUpdateGenotype;
		ne::Genotype _typeSetGenotype;
		ne::Genotype _encoderGenotype;
		ne::Genotype _decoderGenotype;

		int _connectionResponseSize;
		int _nodeOutputSize;
		int _numGases;

		float _inputStrengthScalar;
		float _nodeOutputStrengthScalar;
		float _connectionStrengthScalar;

		std::vector<std::tuple<float, float>> _recurrentNodeInitBounds;
		std::vector<std::tuple<float, float>> _recurrentConnectionInitBounds;

		void setInputOutputCounts(const Field2DEvolverSettings* pSettings, std::mt19937 &generator);

	public:
		void initialize(const Field2DEvolverSettings* pSettings, const std::vector<float> &functionChances, std::mt19937 &generator);
		void crossover(const Field2DEvolverSettings* pSettings, const std::vector<float> &functionChances, const Field2DGenes* pParent1, const Field2DGenes* pParent2, std::mt19937 &generator);
		void mutate(const Field2DEvolverSettings* pSettings, const std::vector<float> &functionChances, std::mt19937 &generator);
		
		static float getSimilarity(const Field2DEvolverSettings* pSettings, const std::vector<float> &functionChances, const Field2DGenes* pGenotype1, const Field2DGenes* pGenotype2, const std::unordered_map<ne::Genotype::FunctionPair, float, ne::Genotype::FunctionPair> &functionFactors);

		const ne::Genotype &getConnectionUpdateGenotype() const {
			return _connectionUpdateGenotype;
		}

		const ne::Genotype &getActivationUpdateGenotype() const {
			return _activationUpdateGenotype;
		}

		const ne::Genotype &getTypeSetGenotype() const {
			return _typeSetGenotype;
		}

		const ne::Genotype &getEncoderGenotype() const {
			return _encoderGenotype;
		}

		const ne::Genotype &getDecoderGenotype() const {
			return _decoderGenotype;
		}

		int getConnectionResponseSize() const {
			return _connectionResponseSize;
		}

		int getNodeOutputSize() const {
			return _nodeOutputSize;
		}

		int getNumGases() const {
			return _numGases;
		}

		float getInputStrengthScalar() const {
			return _inputStrengthScalar;
		}

		float getConnectionStrengthScalar() const {
			return _connectionStrengthScalar;
		}

		float getNodeOutputStrengthScalar() const {
			return _nodeOutputStrengthScalar;
		}

		void readFromStream(std::istream &is);
		void writeToStream(std::ostream &os) const;


		friend class Field2D;
		friend class Field2DCL;
	};
}