#include <erl/platform/ComputeSystem.h>

#include <iostream>

using namespace erl;

void ComputeSystem::create(DeviceType type) {
	std::vector<cl::Platform> allPlatforms;
	cl::Platform::get(&allPlatforms);

	_platform = allPlatforms.front();

	std::vector<cl::Device> allDevices;

	switch (type) {
	case _cpu:
		_platform.getDevices(CL_DEVICE_TYPE_CPU, &allDevices);
		break;
	case _gpu:
		_platform.getDevices(CL_DEVICE_TYPE_GPU, &allDevices);
		break;
	case _both:
		_platform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
		break;
	}

	_device = allDevices.front();

	_context = _device;

	_queue = cl::CommandQueue(_context, _device);
}

void ComputeSystem::create(DeviceType type, Logger &logger) {
	std::vector<cl::Platform> allPlatforms;
	cl::Platform::get(&allPlatforms);

	if (allPlatforms.empty()) {
		logger << " No platforms found. Check your OpenCL installation." << endl;
		return;
	}

	_platform = allPlatforms.front();

	logger << "Using platform: " << _platform.getInfo<CL_PLATFORM_NAME>() << endl;

	std::vector<cl::Device> allDevices;

	switch (type) {
	case _cpu:
		_platform.getDevices(CL_DEVICE_TYPE_CPU, &allDevices);
		break;
	case _gpu:
		_platform.getDevices(CL_DEVICE_TYPE_GPU, &allDevices);
		break;
	case _both:
		_platform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
		break;
	}

	if (allDevices.empty()) {
		logger << " No devices found. Check your OpenCL installation." << endl;
		return;
	}

	_device = allDevices.front();

	logger << "Using device: " << _device.getInfo<CL_DEVICE_NAME>() << endl;

	_context = _device;

	_queue = cl::CommandQueue(_context, _device);
}