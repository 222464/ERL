/*
ERL

Main
*/

#include <erl/ERLConfig.h>

#include <neat/Evolver.h>
#include <neat/NetworkGenotype.h>
#include <neat/NetworkPhenotype.h>
#include <erl/platform/Field2DGenesToCL.h>
#include <erl/visualization/FieldVisualizer.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <fstream>

int main() {
	std::cout << "Welcome to ERL. Version " << ERL_VERSION << std::endl;

	erl::Logger logger;

	logger.createWithFile("erlLog.txt");

	erl::ComputeSystem cs;

	cs.create(logger);

	std::mt19937 generator(time(nullptr));

	neat::EvolverSettings settings;

	std::vector<float> functionChances(3);
	std::vector<std::string> functionNames(3);
	std::vector<std::function<float(float)>> functions(3);
	neat::InnovationNumberType innovNum = 0;

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
	erl::Field2DGenes genes;

	genes.initialize(2, 2, &settings, functionChances, innovNum, generator);

	erl::Field2D field;

	// Load random texture
	sf::Image sfmlImage;

	if (!sfmlImage.loadFromFile("random.bmp")) {
		logger << "Could not load random.bmp!" << erl::endl;
	}

	erl::SoftwareImage2D<sf::Color> softImage;

	softImage.reset(sfmlImage.getSize().x, sfmlImage.getSize().y);

	for (int x = 0; x < sfmlImage.getSize().x; x++)
	for (int y = 0; y < sfmlImage.getSize().y; y++) {
		softImage.setPixel(x, y, sfmlImage.getPixel(x, y));
	}

	std::shared_ptr<cl::Image2D> randomImage(new cl::Image2D(cs.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), softImage.getWidth(), softImage.getHeight(), 0, softImage.getData()));

	field.create(genes, cs, 10, 10, 2, 2, 2, randomImage, functions, functionNames, -1.0f, 1.0f, generator, logger);

	field.update(0.0f, cs, functions, generator);

	erl::FieldVisualizer fv;

	fv.create(cs, "adapter.cl", field, logger);

	fv.update(cs, field);

	sf::Image sfImage;

	sfImage.create(field.getWidth(), field.getHeight());

	for (int x = 0; x < field.getWidth(); x++)
	for (int y = 0; y < field.getHeight(); y++) {
		sfImage.setPixel(x, y, fv.getSoftImage().getPixel(x, y));
	}

	sfImage.saveToFile("result.png");

	system("pause");

	return 0;
}


