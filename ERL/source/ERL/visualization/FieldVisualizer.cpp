#include <erl/visualization/FieldVisualizer.h>

using namespace erl;

bool FieldVisualizer::create(ComputeSystem &cs, const std::string &adapterFileName, const Field2D &field, Logger &logger) {
	cl_int err;
	
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
		logger << "Error building: " << _adapterProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cs.getDevice()) << erl::endl;
		abort();
	}

	//_adapterKernelFunctor = cl::make_kernel<cl::Buffer&, cl::Image2D&, int, int, int>(_adapterProgram, "adapt", &err);

	_adapterKernel = cl::Kernel(_adapterProgram, "adapt", &err);

	if (err != CL_SUCCESS) {
		logger << "Could not create kernel functor!" << endl;
		abort();
	}

	_adaptedSoftImage.reset(field.getWidth(), field.getHeight());

	_adaptedImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), field.getWidth(), field.getHeight());

	return true;
}

void FieldVisualizer::update(ComputeSystem &cs, Field2D &field) {
	//std::vector<cl::Memory> glObjects;
	//glObjects.push_back(_adaptedImage);
	//cs.getQueue().enqueueAcquireGLObjects(&glObjects);

	//_adapterKernelFunctor(cl::EnqueueArgs(cl::NDRange(field.getWidth(), field.getHeight())), field.getBuffer(), _adaptedImage, field.getWidth(), field.getNodeAndConnectionsSize(), field.getNodeOutputSize()).wait();

	_adapterKernel.setArg(0, field.getBuffer());
	_adapterKernel.setArg(1, _adaptedImage);
	_adapterKernel.setArg(2, field.getWidth());
	_adapterKernel.setArg(3, field.getNodeAndConnectionsSize());
	_adapterKernel.setArg(4, field.getNodeOutputSize());

	cs.getQueue().enqueueNDRangeKernel(_adapterKernel, cl::NullRange, cl::NDRange(field.getWidth(), field.getHeight()));

	cl::size_t<3> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	cl::size_t<3> region;
	region[0] = field.getWidth();
	region[1] = field.getHeight();
	region[2] = 1;

	cs.getQueue().enqueueReadImage(_adaptedImage, CL_TRUE, origin, region, 0, 0, _adaptedSoftImage.getData());

	cs.getQueue().finish();

	//cs.getQueue().enqueueReleaseGLObjects(&glObjects);
}