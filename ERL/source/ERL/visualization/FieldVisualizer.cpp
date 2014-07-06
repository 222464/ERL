#include <erl/visualization/FieldVisualizer.h>

using namespace erl;

bool FieldVisualizer::create(ComputeSystem &cs, const std::string &adapterFileName, const Field2D &field, Logger &logger) {
	// Load program
	std::string programStr;

	std::ifstream fromFile(adapterFileName);

	if (!fromFile.is_open()) {
		logger << "Could not open file \"" + adapterFileName + "\" for reading!" << endl;

		return false;
	}

	while (!fromFile.eof() && fromFile.good()) {
		std::string line;

		std::getline(fromFile, line);

		programStr += line + "\n";
	}

	_adapterProgram = cl::Program(cs.getContext(), programStr);

	if (_adapterProgram.build({ cs.getDevice() }) != CL_SUCCESS) {
		logger << " Error building: " << _adapterProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cs.getDevice()) << erl::endl;
		abort();
	}

	_adapterKernelFunctor = cl::make_kernel<cl::Buffer&, cl::ImageGL&, int, int, int>(_adapterProgram, "adapt");

	_adaptedTexture.create(field.getWidth(), field.getHeight());

	sf::Texture::bind(&_adaptedTexture);

	GLint id;

	glGetIntegerv(GL_TEXTURE_BINDING_2D, &id);

	_adaptedImage = cl::ImageGL(cs.getContext(), CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, id);

	return true;
}

void FieldVisualizer::update(Field2D &field) {
	_adapterKernelFunctor(cl::EnqueueArgs(cl::NDRange(field.getWidth(), field.getHeight())), field.getBuffer(), _adaptedImage, field.getWidth(), field.getNodeAndConnectionsSize(), field.getNodeOutputSize());
}