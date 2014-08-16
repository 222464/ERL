#include <erl/field/Field2DEvolver.h>

#include <numeric>
#include <algorithm>

using namespace erl;

Field2DEvolver::Field2DEvolver()
{}

void Field2DEvolver::normalizeFitness(float greedExponent) {
	float minimumFitness = *(std::min_element(_fitnesses.begin(), _fitnesses.end()));

	for (float &f : _fitnesses)
		f = std::pow(f - minimumFitness, greedExponent);
}

size_t Field2DEvolver::roulette(float totalFitness, std::mt19937 &generator) {
	std::uniform_real_distribution<float> cuspDist(0.0f, totalFitness);

	float randomCusp = cuspDist(generator);

	float sumSoFar = 0.0f;

	for (size_t i = 0; i < _fitnesses.size(); i++) {
		sumSoFar += _fitnesses[i];

		if (sumSoFar >= randomCusp)
			return i;
	}

	return 0;
}

void Field2DEvolver::create(size_t populationSize, const Field2DEvolverSettings* pSettings,
	const std::vector<float> &functionChances, std::mt19937 &generator)
{
	_genotypes.resize(populationSize);
	_fitnesses.clear();
	_fitnesses.assign(populationSize, 1.0f);

	for (size_t i = 0; i < populationSize; i++) {
		_genotypes[i].reset(new Field2DGenes());
		_genotypes[i]->initialize(pSettings, functionChances, generator);
	}
}

void Field2DEvolver::epoch(const Field2DEvolverSettings* pSettings, const std::vector<float> &functionChances, std::mt19937 &generator, size_t numElites, float greedExponent, float crossChance) {
	normalizeFitness(greedExponent);

	std::vector<std::shared_ptr<Field2DGenes>> newPopulation;
	newPopulation.reserve(getPopulationSize());

	// Add elites
	std::list<size_t> possibleElites;

	for (size_t i = 0; i < getPopulationSize(); i++)
		possibleElites.push_back(i);

	for (size_t i = 0; i < numElites; i++) {
		// Find highest fitness
		std::list<size_t>::iterator maxIt = possibleElites.begin();

		for (std::list<size_t>::iterator it = possibleElites.begin()++; it != possibleElites.end(); it++)
		if (_fitnesses[*it] > _fitnesses[*maxIt])
			maxIt = it;

		newPopulation.push_back(_genotypes[*maxIt]);

		possibleElites.erase(maxIt);
	}

	float fitnessSum = std::accumulate(_fitnesses.begin(), _fitnesses.end(), 0.0f);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	while (newPopulation.size() < getPopulationSize()) {
		if (dist01(generator) < crossChance) {
			size_t parentIndex1 = roulette(fitnessSum, generator);
			std::shared_ptr<Field2DGenes> newGenotype(new Field2DGenes(*_genotypes[parentIndex1]));

			newGenotype->mutate(pSettings, functionChances, generator);

			newPopulation.push_back(newGenotype);
		}
		else {
			size_t parentIndex1 = roulette(fitnessSum, generator);
			size_t parentIndex2 = roulette(fitnessSum, generator);

			std::shared_ptr<Field2DGenes> newGenotype = std::make_shared<Field2DGenes>();

			newGenotype->crossover(pSettings, functionChances, _genotypes[parentIndex1].get(), _genotypes[parentIndex2].get(), generator);
			newGenotype->mutate(pSettings, functionChances, generator);

			newPopulation.push_back(newGenotype);
		}
	}

	_genotypes = newPopulation;
}