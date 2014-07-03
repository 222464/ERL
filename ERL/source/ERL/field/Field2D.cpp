#include <erl/field/Field2D.h>

using namespace erl;

void Field2D::create(Field2DGenes &genes, ComputeSystem &cs, int width, int height, int connectionRadius, int numInputs, int numOutputs, const std::shared_ptr<cl::Image2D> &randomImage,
	const std::vector<std::function<float(float)>> &activationFunctions, float minRecInit, float maxRecInit, float inputRadius, std::mt19937 &generator) {
	_currentReadBufferIndex = 0;
	_currentWriteBufferIndex = 1;

	_randomImage = randomImage;

	_connectionResponseSize = genes.getConnectionResponseSize();
	_nodeOutputSize = genes.getNodeOutputSize();

	_width = width;
	_height = height;
	_connectionRadius = connectionRadius;

	// Create phenotypes for rules
	_connectionPhenotype.create(genes.getConnectionUpdateGenotype());

	_nodePhenotype.create(genes.getActivationUpdateGenotype());

	// Get rule data
	_connectionPhenotype.getConnectionData(_connectionData);

	_nodePhenotype.getConnectionData(_nodeData);

	_connectionDimensionSize = 2 * _connectionRadius + 1;
	_numConnections = _connectionDimensionSize * _connectionDimensionSize;

	_nodeSize = genes.getNodeOutputSize() + 1 + _nodeData._numRecurrentSourceNodes; // + 1 for type
	_connectionSize = genes.getConnectionResponseSize() + _connectionData._numRecurrentSourceNodes;
	_nodeAndConnectionsSize = _nodeSize + _connectionSize * _numConnections;

	_numNodes = _width * _height;

	_bufferSize = _nodeAndConnectionsSize * _numNodes;

	std::vector<float> buffer(_bufferSize);

	// Type phenotype
	neat::NetworkPhenotype typePhenotype;

	typePhenotype.create(genes.getTypeSetGenotype());

	// Create buffer
	int bufferIndex = 0;

	float widthf = static_cast<float>(_width);
	float heightf = static_cast<float>(_height);

	std::uniform_real_distribution<float> distRecInit(minRecInit, maxRecInit);

	int halfNumNodeRecurrentSources = std::ceil(_nodeData._numRecurrentSourceNodes * 0.5f);

	// Resize init buffers if necessary (add entries)
	while (genes._recurrentNodeInitBounds.size() < _nodeData._numRecurrentSourceNodes) {
		std::tuple<float, float> newBounds = std::make_tuple<float, float>(distRecInit(generator), distRecInit(generator));

		if (std::get<0>(newBounds) > std::get<1>(newBounds))
			std::get<0>(newBounds) = std::get<1>(newBounds) = (std::get<0>(newBounds) +std::get<1>(newBounds)) * 0.5f;

		genes._recurrentNodeInitBounds.push_back(newBounds);
	}

	while (genes._recurrentConnectionInitBounds.size() < _connectionData._numRecurrentSourceNodes) {
		std::tuple<float, float> newBounds = std::make_tuple<float, float>(distRecInit(generator), distRecInit(generator));

		if (std::get<0>(newBounds) > std::get<1>(newBounds))
			std::get<0>(newBounds) = std::get<1>(newBounds) = (std::get<0>(newBounds) +std::get<1>(newBounds)) * 0.5f;

		genes._recurrentConnectionInitBounds.push_back(newBounds);
	}

	for (int ni = 0; ni < _numNodes; ni++) {
		int x = ni % _width;
		int y = ni / _height;

		float xCoord = static_cast<float>(x) / widthf;
		float yCoord = static_cast<float>(y) / heightf;

		for (int oi = 0; oi < genes.getNodeOutputSize(); oi++)
			buffer[bufferIndex++] = 0.0f;

		typePhenotype.getInput(0)._output = xCoord;
		typePhenotype.getInput(1)._output = yCoord;

		typePhenotype.update(activationFunctions);

		buffer[bufferIndex++] = typePhenotype.getOutput(0)._output;

		// Initialize recurrent data
		for (int ri = 0; ri < _nodeData._numRecurrentSourceNodes; ri++) {
			std::uniform_real_distribution<float> distInit(std::get<0>(genes._recurrentNodeInitBounds[ri]), std::get<1>(genes._recurrentNodeInitBounds[ri]));
			buffer[bufferIndex++] = distInit(generator);
		}

		// Connections
		for (int ci = 0; ci < _numConnections; ci++) {
			for (int ri = 0; ri < genes.getConnectionResponseSize(); ri++)
				buffer[bufferIndex++] = 0.0f;

			// Initialize recurrent data
			for (int ri = 0; ri < _connectionData._numRecurrentSourceNodes; ri++) {
				std::uniform_real_distribution<float> distInit(std::get<0>(genes._recurrentConnectionInitBounds[ri]), std::get<1>(genes._recurrentConnectionInitBounds[ri]));
				buffer[bufferIndex++] = distInit(generator);
			}
		}
	}

	// Create type image
	struct IOSet {
		unsigned char _inputIndexPlusOne;
		unsigned char _outputIndexPlusOne;

		IOSet() // 0 means unused
			: _inputIndexPlusOne(0), _outputIndexPlusOne(0)
		{}
	};

	SoftwareImage2D<IOSet> typeSoftwareImage;

	typeSoftwareImage.reset(_width, _height);

	// Set input types (positive index)
	float inputSpacing = static_cast<float>(getHeight()) / static_cast<float>(getNumInputs());

	float startY = inputSpacing * 0.5f;

	unsigned char inputIndex = 0;
	unsigned char outputIndex = 0;

	float xi = inputSpacing * 0.5f;
	int xii = static_cast<int>(xi + 0.5f);

	float xo = inputSpacing * 0.5f;
	int xoi = static_cast<int>(xo + 0.5f);

	for (int i = 0; i < getNumInputs(); i++) {
		float y = startY + i * inputSpacing;
		int yi = static_cast<int>(y + 0.5f);

		// Input
		{
			IOSet ioset;
			ioset._inputIndexPlusOne = inputIndex++;
			ioset._outputIndexPlusOne = 0;

			typeSoftwareImage.setPixel(xii, yi, ioset);
		}

		// Output
		{
			IOSet ioset;
			ioset._inputIndexPlusOne = 0;
			ioset._outputIndexPlusOne = outputIndex++;

			typeSoftwareImage.setPixel(xoi, yi, ioset);
		}
	}

	_typeImage = cl::Image2D(cs.getContext(), CL_MEM_READ_ONLY, cl::ImageFormat(CL_RG, GL_UNSIGNED_BYTE), _width, _height, 0, typeSoftwareImage.getData());

	std::vector<float> outputBuffer(getNumOutputs() * _nodeOutputSize, 0.0f);

	_outputImage = cl::Image1D(cs.getContext(), CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), getNumOutputs(), &outputBuffer[0]);

	// Create OpenCL buffers
	_buffers[0] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _bufferSize * sizeof(float), &buffer[0]);
	_buffers[1] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _bufferSize * sizeof(float), &buffer[1]);

	_kernelFunctor = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Image2D&, cl::Image1D&, cl::Image1D&, cl::Image2D&, RandomSeed, float>(_program, "nodeAdd");

	for (int i = 0; i < numInputs; i++)
		_encoderPhenotypes[i].create(genes.getEncoderGenotype());

	for (int i = 0; i < numOutputs; i++)
		_decoderPhenotypes[i].create(genes.getDecoderGenotype());
}

void Field2D::update(float reward, ComputeSystem &cs, const std::vector<std::function<float(float)>> &activationFunctions, std::mt19937 &generator) {
	std::uniform_real_distribution<float> distSeedX(0.0f, static_cast<float>(_width));
	std::uniform_real_distribution<float> distSeedY(0.0f, static_cast<float>(_height));

	RandomSeed seed;

	seed._x = distSeedX(generator);
	seed._y = distSeedY(generator);

	// Create input image
	std::vector<float> inputBuffer(getNumInputs() * _connectionResponseSize);

	int inputBufferIndex = 0;

	for (int i = 0; i < getNumInputs(); i++) {
		_encoderPhenotypes[i].getInput(0)._output = _inputs[i];

		_encoderPhenotypes[i].update(activationFunctions);

		for (int j = 0; j < _connectionResponseSize; j++)
			inputBuffer[inputBufferIndex++] = _encoderPhenotypes[i].getOutput(j)._output;
	}

	_inputImage = cl::Image1D(cs.getContext(), CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), getNumInputs(), &_inputs[0]);

	// Execute kernel
	_kernelFunctor(cl::EnqueueArgs(_numNodes), _buffers[_currentReadBufferIndex], _buffers[_currentWriteBufferIndex], _typeImage, _inputImage, _outputImage, *_randomImage, seed, reward);

	// Gather outputs
	cl::size_t<3> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	cl::size_t<3> region;
	region[0] = getNumOutputs();
	region[1] = 0;
	region[2] = 0;

	std::vector<float> outputBuffer(getNumOutputs() * _nodeOutputSize);

	cs.getQueue().enqueueReadImage(_outputImage, CL_TRUE, origin, region, 0, 0, &outputBuffer[0]);

	// Decode
	int outputBufferIndex = 0;

	for (int i = 0; i < getNumOutputs(); i++) {
		for (int j = 0; j < _nodeOutputSize; j++)
			_decoderPhenotypes[i].getInput(j)._output = outputBuffer[outputBufferIndex++];

		_decoderPhenotypes[i].update(activationFunctions);

		_outputs[i] = _decoderPhenotypes[i].getOutput(0)._output;
	}
	
	// Swap buffer read/write
	std::swap(_currentReadBufferIndex, _currentWriteBufferIndex);
}