#include <erl/field/Field2DCL.h>

#include <erl/platform/Field2DGenesToCL.h>

using namespace erl;

Field2DCL::Field2DCL()
: _numGasBlurPasses(4)
{}

void Field2DCL::create(Field2DGenes &genes, ComputeSystem &cs, int width, int height, int connectionRadius, int numInputs, int numOutputs,
	int inputRange, int outputRange,
	const std::shared_ptr<cl::Image2D> &randomImage,
	const std::shared_ptr<cl::Program> &gasBlurProgram,
	const std::shared_ptr<cl::Kernel> &gasBlurKernelX,
	const std::shared_ptr<cl::Kernel> &gasBlurKernelY,
	const std::vector<std::function<float(float)>> &activationFunctions, const std::vector<std::string> &activationFunctionNames,
	float minRecInit, float maxRecInit, std::mt19937 &generator,
	Logger &logger)
{
	cl_int err;

	_currentReadBufferIndex = 0;
	_currentWriteBufferIndex = 1;

	_numGases = genes._numGases;
	_typeSize = genes._typeSize;

	_randomImage = randomImage;

	_gasBlurProgram = gasBlurProgram;
	_gasBlurKernelX = gasBlurKernelX;
	_gasBlurKernelY = gasBlurKernelY;

	_connectionResponseSize = genes.getConnectionResponseSize();
	_nodeOutputSize = genes.getNodeOutputSize();

	_width = width;
	_height = height;
	_connectionRadius = connectionRadius;

	_inputs.clear();
	_inputs.assign(numInputs, 0.0f);

	_outputs.clear();
	_outputs.assign(numOutputs, 0.0f);

	_inputStrengthScalar = genes.getInputStrengthScalar();
	_connectionStrengthScalar = genes.getConnectionStrengthScalar();
	_nodeOutputStrengthScalar = genes.getNodeOutputStrengthScalar();

	_numOutputsPerBlob = (outputRange + 1) * (outputRange + 1);

	// Create phenotypes for rules
	_connectionPhenotype.createFromGenotype(genes.getConnectionUpdateGenotype());

	_nodePhenotype.createFromGenotype(genes.getActivationUpdateGenotype());

	_connectionDimensionSize = 2 * _connectionRadius + 1;
	_numConnections = _connectionDimensionSize * _connectionDimensionSize;

	_nodeSize = genes.getNodeOutputSize() + _typeSize + _nodePhenotype.getRecurrentDataSize();
	_connectionSize = _connectionPhenotype.getRecurrentDataSize();
	_nodeAndConnectionsSize = _nodeSize + _connectionSize * _numConnections;

	_numNodes = _width * _height;

	_bufferSize = _nodeAndConnectionsSize * _numNodes;

	std::vector<float> buffer(_bufferSize);

	// Type phenotype
	ne::Phenotype typePhenotype;

	typePhenotype.createFromGenotype(genes.getTypeSetGenotype());

	// Create buffer
	int bufferIndex = 0;

	float widthf = static_cast<float>(_width);
	float heightf = static_cast<float>(_height);

	std::uniform_real_distribution<float> distRecInit(minRecInit, maxRecInit);

	// Resize init buffers if necessary (add entries)
	while (genes._recurrentNodeInitBounds.size() < _nodePhenotype.getRecurrentDataSize()) {
		std::tuple<float, float> newBounds = std::make_tuple<float, float>(distRecInit(generator), distRecInit(generator));

		if (std::get<0>(newBounds) > std::get<1>(newBounds))
			std::get<0>(newBounds) = std::get<1>(newBounds) = (std::get<0>(newBounds) + std::get<1>(newBounds)) * 0.5f;

		genes._recurrentNodeInitBounds.push_back(newBounds);
	}

	while (genes._recurrentConnectionInitBounds.size() < _connectionPhenotype.getRecurrentDataSize()) {
		std::tuple<float, float> newBounds = std::make_tuple<float, float>(distRecInit(generator), distRecInit(generator));

		if (std::get<0>(newBounds) > std::get<1>(newBounds))
			std::get<0>(newBounds) = std::get<1>(newBounds) = (std::get<0>(newBounds) + std::get<1>(newBounds)) * 0.5f;

		genes._recurrentConnectionInitBounds.push_back(newBounds);
	}

	std::vector<float> typeSetRecurrentData(typePhenotype.getRecurrentDataSize(), 0.0f);

	for (int ni = 0; ni < _numNodes; ni++) {
		int x = ni % _width;
		int y = ni / _width;

		float xCoord = static_cast<float>(x) / widthf;
		float yCoord = static_cast<float>(y) / heightf;

		for (int oi = 0; oi < genes.getNodeOutputSize(); oi++)
			buffer[bufferIndex++] = 0.0f;

		std::vector<float> inputs(2);
		std::vector<float> outputs(_typeSize);

		inputs[0] = xCoord;
		inputs[1] = yCoord;
		
		typePhenotype.execute(inputs, outputs, typeSetRecurrentData, activationFunctions);

		for (int ti = 0; ti < _typeSize; ti++)
			buffer[bufferIndex++] = outputs[ti];

		// Initialize recurrent data
		for (int ri = 0; ri < _nodePhenotype.getRecurrentDataSize(); ri++) {
			std::uniform_real_distribution<float> distInit(std::get<0>(genes._recurrentNodeInitBounds[ri]), std::get<1>(genes._recurrentNodeInitBounds[ri]));
			buffer[bufferIndex++] = distInit(generator);
		}

		// Connections
		for (int ci = 0; ci < _numConnections; ci++) {
			// Initialize recurrent data
			for (int ri = 0; ri < _connectionPhenotype.getRecurrentDataSize(); ri++) {
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
	float inputStartY = inputSpacing * 0.5f;

	float outputSpacing = static_cast<float>(getHeight()) / static_cast<float>(getNumOutputs());
	float outputStartY = outputSpacing * 0.5f;

	unsigned char inputIndex = 0;
	unsigned char outputIndex = 0;

	float xi = inputSpacing;
	int xii = static_cast<int>(xi + 0.5f);

	float xo = getWidth() - outputSpacing;
	int xoi = static_cast<int>(xo + 0.5f);

	for (int i = 0; i < getNumInputs(); i++) {
		float y = inputStartY + i * inputSpacing;
		int yi = static_cast<int>(y + 0.5f);

		IOSet ioset;
		ioset._inputIndexPlusOne = inputIndex++ + 1;
		ioset._outputIndexPlusOne = 0;

		// Box
		for (int dx = -inputRange; dx <= inputRange; dx++)
		for (int dy = -inputRange; dy <= inputRange; dy++) {
			int nx = (xii + dx) % _width;
			int ny = (yi + dy) % _height;
			nx = nx < 0 ? nx + _width : nx;
			ny = ny < 0 ? ny + _height : ny;

			typeSoftwareImage.setPixel(nx, ny, ioset);
		}
	}

	for (int i = 0; i < getNumOutputs(); i++) {
		float y = outputStartY + i * outputSpacing;
		int yi = static_cast<int>(y + 0.5f);

		// Box
		for (int dx = -inputRange; dx <= inputRange; dx++)
		for (int dy = -inputRange; dy <= inputRange; dy++) {
			int nx = (xoi + dx) % _width;
			int ny = (yi + dy) % _height;
			nx = nx < 0 ? nx + _width : nx;
			ny = ny < 0 ? ny + _height : ny;

			IOSet ioset;
			ioset._inputIndexPlusOne = 0;
			ioset._outputIndexPlusOne = outputIndex++ + 1;

			typeSoftwareImage.setPixel(nx, ny, ioset);
		}
	}

	_typeImage = cl::Image2D(cs.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_RG, CL_UNSIGNED_INT8), _width, _height, 0, typeSoftwareImage.getData());

	// Create OpenCL buffers
	_buffers[0] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buffer.size() * sizeof(float), &buffer[0]);
	_buffers[1] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buffer.size() * sizeof(float), &buffer[0]);

	std::vector<float> gasInit(_numNodes * _numGases, 0.0f);

	_gasBuffers[0] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gasInit.size() * sizeof(float), &gasInit[0]);
	_gasBuffers[1] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, gasInit.size() * sizeof(float), &gasInit[0]);
	
	std::vector<float> outputBuffer(getNumOutputs() * _nodeOutputSize * _numOutputsPerBlob, 0.0f);

	_outputImage = cl::Image1D(cs.getContext(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_R, CL_FLOAT), outputBuffer.size(), &outputBuffer[0]);

	_program = cl::Program(cs.getContext(), field2DGenesNodeUpdateToCL(genes, *this, _connectionPhenotype, _nodePhenotype, activationFunctionNames, _width, _height, _connectionRadius, numInputs, numOutputs));

	if (_program.build(std::vector<cl::Device>(1, cs.getDevice())) != CL_SUCCESS) {
		logger << "Error building: " << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cs.getDevice()) << erl::endl;
		abort();
	}

	//_kernelFunctor = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Image2D&, cl::Image1D&, cl::Image1D&, cl::Image2D&, RandomSeed, float>(_program, "nodeUpdate");

	_kernel = cl::Kernel(_program, "nodeUpdate");

	_encoderPhenotypes.resize(numInputs);
	_decoderPhenotypes.resize(numOutputs);
	_encoderRecurrentData.resize(numInputs);
	_decoderRecurrentData.resize(numOutputs);

	for (int i = 0; i < numInputs; i++) {
		_encoderPhenotypes[i].createFromGenotype(genes.getEncoderGenotype());

		_encoderRecurrentData[i].assign(_encoderPhenotypes[i].getRecurrentDataSize(), 0.0f);
	}

	for (int i = 0; i < numOutputs; i++) {
		_decoderPhenotypes[i].createFromGenotype(genes.getDecoderGenotype());

		_decoderRecurrentData[i].assign(_decoderPhenotypes[i].getRecurrentDataSize(), 0.0f);
	}
}

void Field2DCL::update(float reward, ComputeSystem &cs, const std::vector<std::function<float(float)>> &activationFunctions, int substeps, std::mt19937 &generator) {
	std::uniform_real_distribution<float> distSeedX(0.0f, static_cast<float>(_width));
	std::uniform_real_distribution<float> distSeedY(0.0f, static_cast<float>(_height));

	// Create input image
	std::vector<float> inputBuffer(getNumInputs() * _connectionResponseSize);
	
	int inputBufferIndex = 0;

	for (int i = 0; i < getNumInputs(); i++) {
		std::vector<float> outputBuffer(_encoderPhenotypes[i].getNumOutputs(), 0.0f);

		_encoderPhenotypes[i].execute(std::vector<float>(1, _inputs[i]), outputBuffer, _encoderRecurrentData[i], activationFunctions);

		for (int j = 0; j < _connectionResponseSize; j++)
			inputBuffer[inputBufferIndex++] = outputBuffer[j] * _inputStrengthScalar;
	}

 	_inputImage = cl::Image1D(cs.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_R, CL_FLOAT), inputBuffer.size(), &inputBuffer[0]);

	// Execute kernel
	//_kernelFunctor(cl::EnqueueArgs(cl::NDRange(_numNodes)), _buffers[_currentReadBufferIndex], _buffers[_currentWriteBufferIndex], _typeImage, _inputImage, _outputImage, *_randomImage, seed, reward).wait();
	for (int s = 0; s < substeps; s++) {
		RandomSeed seed;
		
		seed._x = distSeedX(generator);
		seed._y = distSeedY(generator);

		_kernel.setArg(0, _buffers[_currentReadBufferIndex]);
		_kernel.setArg(1, _gasBuffers[_currentReadBufferIndex]);
		_kernel.setArg(2, _buffers[_currentWriteBufferIndex]);
		_kernel.setArg(3, _gasBuffers[_currentWriteBufferIndex]);
		_kernel.setArg(4, _typeImage);
		_kernel.setArg(5, _inputImage);
		_kernel.setArg(6, _outputImage);
		_kernel.setArg(7, *_randomImage);
		_kernel.setArg(8, seed);
		_kernel.setArg(9, reward);

		cl::Event event;

		cs.getQueue().enqueueNDRangeKernel(_kernel, cl::NullRange, cl::NDRange(_width, _height), cl::NullRange, nullptr, &event);

		event.wait();

		// Swap buffer read/write
		std::swap(_currentReadBufferIndex, _currentWriteBufferIndex);
	}

	// Gather outputs
	cl::size_t<3> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	cl::size_t<3> region;
	region[0] = getNumOutputs() * _nodeOutputSize * _numOutputsPerBlob;
	region[1] = 1;
	region[2] = 1;

	std::vector<float> outputBuffer(getNumOutputs() * _nodeOutputSize * _numOutputsPerBlob);

	cl::Event event;

	cs.getQueue().enqueueReadImage(_outputImage, CL_TRUE, origin, region, 0, 0, &outputBuffer[0], nullptr, &event);

	event.wait();

	// Decode
	int outputBufferIndex = 0;

	float numOutputsPerBlobInv = 1.0f / _numOutputsPerBlob;

	for (int i = 0; i < getNumOutputs(); i++) {
		std::vector<float> decoderInputs(_nodeOutputSize, 0.0f);

		for (int j = 0; j < _numOutputsPerBlob; j++)
		for (int k = 0; k < _nodeOutputSize; k++)
			decoderInputs[k] += outputBuffer[outputBufferIndex++];

		for (int k = 0; k < _nodeOutputSize; k++)
			decoderInputs[k] *= numOutputsPerBlobInv;

		std::vector<float> outputBuffer(1, 0.0f);

		_decoderPhenotypes[i].execute(decoderInputs, outputBuffer, _decoderRecurrentData[i], activationFunctions);

		_outputs[i] = outputBuffer[0];
	}

	// Blur gas
	unsigned char _currentBlurReadBufferIndex = _currentWriteBufferIndex;
	unsigned char _currentBlurWriteBufferIndex = _currentReadBufferIndex;

	for (int i = 0; i < _numGasBlurPasses; i++) {
		{
			_gasBlurKernelX->setArg(0, _gasBuffers[_currentBlurReadBufferIndex]);
			_gasBlurKernelX->setArg(1, _gasBuffers[_currentBlurWriteBufferIndex]);
			_gasBlurKernelX->setArg(2, getWidth());
			_gasBlurKernelX->setArg(3, getHeight());
			_gasBlurKernelX->setArg(4, getNumNodes());

			cl::Event event;

			cs.getQueue().enqueueNDRangeKernel(*_gasBlurKernelX, cl::NullRange, cl::NDRange(getWidth(), getHeight(), getNumGases()), cl::NullRange, nullptr, &event);

			event.wait();

			std::swap(_currentBlurReadBufferIndex, _currentBlurWriteBufferIndex);
		}

		{
			_gasBlurKernelY->setArg(0, _gasBuffers[_currentBlurReadBufferIndex]);
			_gasBlurKernelY->setArg(1, _gasBuffers[_currentBlurWriteBufferIndex]);
			_gasBlurKernelY->setArg(2, getWidth());
			_gasBlurKernelY->setArg(3, getHeight());
			_gasBlurKernelY->setArg(4, getNumNodes());

			cl::Event event;

			cs.getQueue().enqueueNDRangeKernel(*_gasBlurKernelY, cl::NullRange, cl::NDRange(getWidth(), getHeight(), getNumGases()), cl::NullRange, nullptr, &event);

			event.wait();

			std::swap(_currentBlurReadBufferIndex, _currentBlurWriteBufferIndex);
		}
	}
}