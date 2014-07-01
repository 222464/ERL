#include <erl/field/Field2D.h>

using namespace erl;

void Field2D::create(Field2DGenes &genes, ComputeSystem &cs, int width, int height, int connectionRadius, int numInputs, int numOutputs,
	const std::vector<std::function<float(float)>> &activationFunctions, float minRecInit, float maxRecInit, float inputRadius, std::mt19937 &generator) {
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

	_nodeSize = genes.getNodeOutputSize() + 3 + _nodeData._numRecurrentSourceNodes; // + 3 for type, input index, and input strength
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
			std::get<0>(newBounds) = std::get<1>(newBounds) = (std::get<0>(newBounds) + std::get<1>(newBounds)) * 0.5f;

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

		// Set input index and strength to -1 for now, may change later
		buffer[bufferIndex++] = -1.0f;
		buffer[bufferIndex++] = -1.0f;

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

	// Set inputs

	// Create OpenCL buffers
	_buffers[0] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _bufferSize * sizeof(float), &buffer[0]);
	_buffers[1] = cl::Buffer(cs.getContext(), CL_MEM_READ_WRITE, _bufferSize * sizeof(float), &buffer[0]);
}

void Field2D::update(ComputeSystem &cs) {

}