/*
ERL

Main
*/

#include <erl/ERLConfig.h>

#include <neat/Evolver.h>
#include <neat/NetworkGenotype.h>
#include <neat/NetworkPhenotype.h>
#include <neat/Evolver.h>

#include <erl/platform/Field2DGenesToCL.h>
#include <erl/visualization/FieldVisualizer.h>
#include <erl/simulation/EvolutionaryTrainer.h>
#include <erl/field/Field2DEvolverSettings.h>

#include <erl/experiments/ExperimentAND.h>
#include <erl/experiments/ExperimentOR.h>
#include <erl/experiments/ExperimentXOR.h>
#include <erl/experiments/ExperimentPoleBalancing.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <fstream>

// Sets the mode of execution
//#define TRAIN_ERL

int main() {
	std::cout << "Welcome to ERL. Version " << ERL_VERSION << std::endl;

	erl::Logger logger;

	logger.createWithFile("erlLog.txt");

	erl::ComputeSystem cs;

	cs.create(erl::ComputeSystem::_gpu, logger);

	std::mt19937 generator(time(nullptr));

	std::vector<float> functionChances(3);
	std::vector<std::string> functionNames(3);
	std::vector<std::function<float(float)>> functions(3);

	functionChances[0] = 1.0f;
	functionChances[1] = 1.0f;
	functionChances[2] = 1.0f;

	functionNames[0] = "sigmoid";
	functionNames[1] = "sin";
	functionNames[2] = "linear";

	functions[0] = std::bind(neat::Neuron::sigmoid, std::placeholders::_1);
	functions[1] = std::bind(std::sinf, std::placeholders::_1);
	functions[2] = std::bind([](float x) { return std::min<float>(2.0f, std::max<float>(-2.0f, x)); }, std::placeholders::_1);

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

	std::shared_ptr<cl::Image2D> randomImage(new cl::Image2D(cs.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_RGBA, CL_UNORM_INT8), softImage.getWidth(), softImage.getHeight(), 0, softImage.getData()));

	// Read source
	std::ifstream is("gasBlur.cl");

	if (!is.is_open()) {
		logger << "Could not open gas blur kernel!" << erl::endl;
		abort();
	}

	std::string blurSource = "";

	while (!is.eof() && is.good()) {
		std::string line;
		std::getline(is, line);

		blurSource += line + "\n";
	}

	std::shared_ptr<cl::Program> blurProgram(new cl::Program(cs.getContext(), blurSource));

	if (blurProgram->build({ cs.getDevice() }) != CL_SUCCESS) {
		logger << "Error building: " << blurProgram->getBuildInfo<CL_PROGRAM_BUILD_LOG>(cs.getDevice()) << erl::endl;
		abort();
	}

	std::shared_ptr<cl::Kernel> blurKernelX(new cl::Kernel(*blurProgram, "blurX"));
	std::shared_ptr<cl::Kernel> blurKernelY(new cl::Kernel(*blurProgram, "blurY"));

#ifdef TRAIN_ERL
	// ------------------------------------------- Training -------------------------------------------

	std::shared_ptr<neat::EvolverSettings> settings(new erl::Field2DEvolverSettings());

	erl::EvolutionaryTrainer trainer;

	trainer.create(functionChances, settings, randomImage, blurProgram, blurKernelX, blurKernelY, functions, functionNames, -1.0f, 1.0f, generator);

	trainer.addExperiment(std::shared_ptr<erl::Experiment>(new ExperimentPoleBalancing()));
	//trainer.addExperiment(std::shared_ptr<erl::Experiment>(new ExperimentOR()));
	//trainer.addExperiment(std::shared_ptr<erl::Experiment>(new ExperimentAND()));
	//trainer.addExperiment(std::shared_ptr<erl::Experiment>(new ExperimentXOR()));

	for (size_t g = 0; g < 10000; g++) {
		logger << "Evaluating generation " << std::to_string(g + 1) << "." << erl::endl;

		trainer.evaluate(cs, logger, generator);

		logger << "Reproducing generation " << std::to_string(g + 1) << "." << erl::endl;

		trainer.reproduce(generator);

		logger << "Saving best to \"erl1.txt\"" << erl::endl;

		std::ofstream toFile("erlBestResultSoFar.txt");

		trainer.writeBestToStream(toFile);

		toFile.close();

		logger << "Generation completed." << erl::endl;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			break;
	}

#else

	// ------------------------------------------- Testing -------------------------------------------

	std::shared_ptr<neat::EvolverSettings> settings(new erl::Field2DEvolverSettings());

	erl::Field2DGenes genes;

	//std::ifstream fromFile("erlBestResultSoFar.txt");

	//genes.readFromStream(fromFile);

	//fromFile.close();

	neat::InnovationNumberType innovNum;

	genes.initialize(2, 1, settings.get(), functionChances, innovNum, generator);

	erl::Field2D field;

	float sizeScalar = 800.0f / 400.0f;

	field.create(genes, cs, 400, 400, 4, 2, 1, randomImage, blurProgram, blurKernelX, blurKernelY, functions, functionNames, -1.0f, 1.0f, generator, logger);

	field.setInput(0, 10.0f);
	field.setInput(1, 10.0f);

	sf::RenderWindow window;
	window.create(sf::VideoMode(800, 800), "ERL Test", sf::Style::Default);

	window.setVerticalSyncEnabled(true);

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	erl::FieldVisualizer fv;

	fv.create(cs, "adapter.cl", field, logger);

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent)) {
			switch (windowEvent.type) {
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		float reward = 0.0f;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			reward = 1.0f;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			reward = -1.0f;

		// -------------------------------------------------------------------

		window.clear();

		field.update(reward, cs, functions, 1, generator);

		fv.update(cs, field);

		sf::Image image;
		image.create(fv.getSoftImage().getWidth(), fv.getSoftImage().getHeight());

		for (int x = 0; x < image.getSize().x; x++)
		for (int y = 0; y < image.getSize().y; y++) {
			image.setPixel(x, y, fv.getSoftImage().getPixel(x, y));
		}

		sf::Texture texture;
		texture.loadFromImage(image);

		sf::Sprite sprite;
		sprite.setTexture(texture);
		sprite.setScale(sf::Vector2f(sizeScalar, sizeScalar));

		window.draw(sprite);

		// -------------------------------------------------------------------

		window.display();

		dt = clock.getElapsedTime().asSeconds();
	} while (!quit);

#endif

	return 0;
}


