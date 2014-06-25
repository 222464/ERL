/*
ERL

Logger
*/

#pragma once

#include <iostream>
#include <fstream>
#include <string>

namespace erl {
	const std::string endl = "\n";

	class Logger {
	private:
		std::ofstream _logFileStream;

		bool _showInConsole;

	public:
		~Logger() {
			close();
		}

		void create(bool showInConsole = true);
		void create(const std::string &logFileName, bool showInConsole = true);
		
		void close();

		Logger &operator<<(const std::string &str);
	};
}