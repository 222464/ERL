/*
ERL

Platform
*/

#pragma once

#include <erl/platform/Logger.h>
#include <CL/cl.hpp>

namespace erl {
	class ComputeSystem {
	private:
		cl::Platform _platform;
		cl::Device _device;
		cl::Context _context;
		cl::CommandQueue _queue;

	public:
		void create();
		void create(Logger &logger);

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