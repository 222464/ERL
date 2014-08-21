/*
ERL

Platform
*/

#pragma once

#include <erl/platform/Logger.h>
#include <erl/platform/Uncopyable.h>
#include <CL/cl.hpp>

namespace erl {
	class ComputeSystem : public Uncopyable {
	public:
		enum DeviceType {
			_cpu, _gpu, _both
		};

	private:
		cl::Platform _platform;
		cl::Device _device;
		cl::Context _context;
		cl::CommandQueue _queue;

	public:
		void create(DeviceType type);
		void create(DeviceType type, Logger &logger);

		cl::Platform &getPlatform() {
			return _platform;
		}

		cl::Device &getDevice() {
			return _device;
		}

		cl::Context &getContext() {
			return _context;
		}

		cl::CommandQueue &getQueue() {
			return _queue;
		}
	};
}