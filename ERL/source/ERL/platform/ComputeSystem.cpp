#include <erl/platform/ComputeSystem.h>

#include <iostream>

using namespace erl;

void ComputeSystem::create() {
	std::vector<cl::Platform> allPlatforms;
	cl::Platform::get(&allPlatforms);

	_platform = allPlatforms.front();

	std::vector<cl::Device> allDevices;

	_platform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);

	_device = allDevices.front();

	_context = _device;
}

void ComputeSystem::create(Logger &logger) {
	std::vector<cl::Platform> allPlatforms;
	cl::Platform::get(&allPlatforms);

	if (allPlatforms.empty()) {
		logger << " No platforms found. Check your OpenCL installation." << endl;
		return;
	}

	_platform = allPlatforms.front();

	logger << "Using platform: " << _platform.getInfo<CL_PLATFORM_NAME>() << endl;

	std::vector<cl::Device> allDevices;

	_platform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);

	if (allDevices.empty()) {
		logger << " No devices found. Check your OpenCL installation." << endl;
		return;
	}

	_device = allDevices.front();

	logger << "Using device: " << _device.getInfo<CL_DEVICE_NAME>() << endl;

	_context = _device;
}