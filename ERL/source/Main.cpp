/*
ERL

Main
*/

#include <erl/ERLConfig.h>

#include <neat/Evolver.h>
#include <neat/NetworkGenotype.h>
#include <neat/NetworkPhenotype.h>
#include <erl/platform/RuleToCL.h>

#include <time.h>
#include <iostream>
#include <fstream>

int main() {
	std::cout << "Welcome to ERL. Version " << ERL_VERSION << std::endl;

	std::mt19937 generator(time(nullptr));

	neat::EvolverSettings settings;

	std::vector<float> functionChances(3);
	std::vector<std::string> functionNames(3);
	std::vector<std::function<float(float)>> functions(3);

	functionChances[0] = 1.0f;
	functionChances[1] = 1.0f;
	functionChances[2] = 1.0f;

	functionNames[0] = "sigmoid";
	functionNames[1] = "sin";
	functionNames[2] = "exp";

	functions[0] = std::bind(neat::Neuron::sigmoid, std::placeholders::_1);
	functions[1] = std::bind(std::sinf, std::placeholders::_1);
	functions[2] = std::bind(std::expf, std::placeholders::_1);

	// Generate random genotype
	neat::NetworkGenotype gt;
	neat::InnovationNumberType innovNum;

	gt.initialize(2, 2, &settings, functionChances, innovNum, generator);

	for (size_t i = 0; i < 60; i++)
		gt.mutate(&settings, functionChances, innovNum, generator);

	neat::NetworkPhenotype pt;

	pt.create(gt);

	std::ofstream toFile("testOutput.txt");

	toFile << erl::ruleToCL(pt, "testRule1", "sdasd", functionNames);

	toFile.close();

	system("pause");

	return 0;
}


