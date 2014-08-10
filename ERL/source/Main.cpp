/*
ERL

Main
*/

#include <erl/ERLConfig.h>

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
#define TRAIN_ERL
//#define VISUALIZE_ERL
//#define EXPERIMENT_ERL

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

	functions[0] = std::bind([](float x) { return 1.0f / (1.0f + std::exp(-x)); }, std::placeholders::_1);
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

	std::shared_ptr<erl::Field2DEvolverSettings> settings(new erl::Field2DEvolverSettings());

	erl::EvolutionaryTrainer trainer;

	trainer.create(5, settings.get(), functionChances, randomImage, blurProgram, blurKernelX, blurKernelY, functions, functionNames, -1.0f, 1.0f, generator);

	trainer.addExperiment(std::shared_ptr<erl::Experiment>(new ExperimentPoleBalancing()));
	//trainer.addExperiment(std::shared_ptr<erl::Experiment>(new ExperimentOR()));
	//trainer.addExperiment(std::shared_ptr<erl::Experiment>(new ExperimentAND()));
	//trainer.addExperiment(std::shared_ptr<erl::Experiment>(new ExperimentXOR()));

	for (size_t g = 0; g < 10000; g++) {
		logger << "Evaluating generation " << std::to_string(g + 1) << "." << erl::endl;

		trainer.evaluate(settings.get(), functionChances, cs, logger, generator);

		logger << "Saving best to \"erlBestResultSoFar.txt\"" << erl::endl;

		std::ofstream toFile("erlBestResultSoFar.txt");

		trainer.writeBestToStream(toFile);

		logger << "Reproducing generation " << std::to_string(g + 1) << "." << erl::endl;

		trainer.reproduce(settings.get(), functionChances, generator);

		toFile.close();

		logger << "Generation completed." << erl::endl;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			break;
	}

#else
#ifdef VISUALIZE_ERL
	// ------------------------------------------- Testing -------------------------------------------

	std::shared_ptr<erl::Field2DEvolverSettings> settings(new erl::Field2DEvolverSettings());

	erl::Field2DGenes genes;

	std::ifstream fromFile("erlBestResultSoFar.txt");

	genes.readFromStream(fromFile);

	fromFile.close();

	//neat::InnovationNumberType innovNum;

	//genes.initialize(2, 1, settings.get(), functionChances, innovNum, generator);

	erl::Field2DCL field;

	float sizeScalar = 800.0f / 64.0f;

	field.create(genes, cs, 32, 32, 2, 4, 1, 1, 1, randomImage, blurProgram, blurKernelX, blurKernelY, functions, functionNames, -1.0f, 1.0f, generator, logger);

	field.setInput(0, 1.0f);
	field.setInput(1, -1.0f);

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

		field.update(reward, cs, functions, 16, generator);

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
#elif defined (EXPERIMENT_ERL)
	// ------------------------------------------- Testing -------------------------------------------

	std::shared_ptr<erl::Field2DEvolverSettings> settings(new erl::Field2DEvolverSettings());

	erl::Field2DGenes genes;

	std::ifstream fromFile("erlBestResultSoFar.txt");

	genes.readFromStream(fromFile);

	fromFile.close();

	//neat::InnovationNumberType innovNum;

	//genes.initialize(2, 1, settings.get(), functionChances, innovNum, generator);

	//for (int i = 0; i < 20; i++)
	//	genes.mutate(settings.get(), functionChances, innovNum, generator);

	ExperimentPoleBalancing ex;
	float exRes = ex.evaluate(genes, settings.get(), randomImage, blurProgram, blurKernelX, blurKernelY, functions, functionNames, -1.0f, 1.0f, logger, cs, generator);

	std::cout << "Experiment result: " << exRes << std::endl;

	erl::Field2DCL field;

	float sizeScalar = 600.0f / 32.0f;

	field.create(genes, cs, 32, 32, 2, 4, 1, 1, 1, randomImage, blurProgram, blurKernelX, blurKernelY, functions, functionNames, -1.0f, 1.0f, generator, logger);

	sf::RenderWindow window;
	window.create(sf::VideoMode(1400, 600), "ERL Test", sf::Style::Default);

	window.setVerticalSyncEnabled(true);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;
	sf::Texture poleTexture;

	backgroundTexture.loadFromFile("resources/background.png");
	cartTexture.loadFromFile("resources/cart.png");
	poleTexture.loadFromFile("resources/pole.png");

	// --------------------------------------------------------------------

	sf::Sprite backgroundSprite;
	sf::Sprite cartSprite;
	sf::Sprite poleSprite;

	backgroundSprite.setTexture(backgroundTexture);
	cartSprite.setTexture(cartTexture);
	poleSprite.setTexture(poleTexture);

	backgroundSprite.setPosition(sf::Vector2f(0.0f, 0.0f));

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y)));
	poleSprite.setOrigin(sf::Vector2f(static_cast<float>(poleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(poleSprite.getTexture()->getSize().y)));

	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 20.0f;
	float cartMass = 2.0f;
	sf::Vector2f massPos(0.0f, poleLength);
	sf::Vector2f massVel(0.0f, 0.0f);
	float poleAngle = static_cast<float>(std::_Pi) * 0.0f;
	float poleAngleVel = 0.0f;
	float poleAngleAccel = 0.0f;
	float cartX = 0.0f;
	float cartVelX = 0.0f;
	float cartAccelX = 0.0f;
	float poleRotationalFriction = 0.008f;
	float cartMoveRadius = 1.8f;
	float cartFriction = 0.02f;
	float maxSpeed = 3.0f;

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;

	float prevFitness = 0.0f;

	bool reverseDirection = false;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	erl::FieldVisualizer fv;

	fv.create(cs, "adapter.cl", field, logger);

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// Update fitness
		if (poleAngle < static_cast<float>(std::_Pi))
			fitness = -(static_cast<float>(std::_Pi) * 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(std::_Pi) * 0.5f - (static_cast<float>(std::_Pi) * 2.0f - poleAngle));

		//fitness = fitness - std::fabsf(poleAngleVel * 40.0f);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			fitness = -cartX;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			fitness = cartX;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		float error = dFitness * 10.0f;

		//agent.reinforceArp(std::min(1.0f, std::max(-1.0f, error)) * 0.5f + 0.5f, 0.1f, 0.05f);

		field.setInput(0, cartX * 0.25f);
		field.setInput(1, cartVelX);
		field.setInput(2, std::fmodf(poleAngle + static_cast<float>(std::_Pi), 2.0f * static_cast<float>(std::_Pi)));
		field.setInput(3, poleAngleVel);

		field.update(error, cs, functions, 16, generator);

		float dir = std::min<float>(1.0f, std::max<float>(-1.0f, field.getOutput(0)));

		//dir = 1.4f * (dir * 2.0f - 1.0f);

		float agentForce = 4000.0f * dir;
		//float agentForce = 2000.0f * agent.getOutput(0);

		// ---------------------------- Physics ----------------------------

		float pendulumCartAccelX = cartAccelX;

		if (cartX < -cartMoveRadius)
			pendulumCartAccelX = 0.0f;
		else if (cartX > cartMoveRadius)
			pendulumCartAccelX = 0.0f;

		poleAngleAccel = pendulumCartAccelX * std::cosf(poleAngle) + g * std::sinf(poleAngle);
		poleAngleVel += -poleRotationalFriction * poleAngleVel + poleAngleAccel * dt;
		poleAngle += poleAngleVel * dt;

		massPos = sf::Vector2f(cartX + std::cosf(poleAngle + static_cast<float>(std::_Pi) * 0.5f) * poleLength, std::sinf(poleAngle + static_cast<float>(std::_Pi) * 0.5f) * poleLength);

		float force = 0.0f;

		if (std::fabsf(cartVelX) < maxSpeed) {
			force = std::max<float>(-4000.0f, std::min<float>(4000.0f, agentForce));

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
				force = -4000.0f;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
				force = 4000.0f;
		}

		if (cartX < -cartMoveRadius) {
			cartX = -cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}
		else if (cartX > cartMoveRadius) {
			cartX = cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}

		cartAccelX = 0.25f * (force + massMass * poleLength * poleAngleAccel * std::cosf(poleAngle) - massMass * poleLength * poleAngleVel * poleAngleVel * std::sinf(poleAngle)) / (massMass + cartMass);
		cartVelX += -cartFriction * cartVelX + cartAccelX * dt;
		cartX += cartVelX * dt;

		poleAngle = std::fmodf(poleAngle, (2.0f * static_cast<float>(std::_Pi)));

		if (poleAngle < 0.0f)
			poleAngle += static_cast<float>(std::_Pi) * 2.0f;

		// ---------------------------- Rendering ----------------------------

		window.clear();

		window.draw(backgroundSprite);

		cartSprite.setPosition(sf::Vector2f(static_cast<float>(800.0f) * 0.5f + pixelsPerMeter * cartX, static_cast<float>(600.0f) * 0.5f + 3.0f));

		window.draw(cartSprite);

		poleSprite.setPosition(cartSprite.getPosition() + sf::Vector2f(0.0f, -45.0f));
		poleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(std::_Pi) + 180.0f);

		window.draw(poleSprite);

		// ------------------------

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

		sprite.setPosition(800.0f, 0.0f);

		window.draw(sprite);

		// -------------------------------------------------------------------

		window.display();

		dt = clock.getElapsedTime().asSeconds();
	} while (!quit);
#endif
#endif

	return 0;
}


