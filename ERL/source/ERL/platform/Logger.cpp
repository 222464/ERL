#include <erl/platform/Logger.h>

using namespace erl;

void Logger::create(bool showInConsole) {
	_showInConsole = showInConsole;
}

void Logger::createWithFile(const std::string &logFileName, bool showInConsole) {
	_logFileStream.open(logFileName);

	_showInConsole = showInConsole;
}

void Logger::close() {
	if (_logFileStream.is_open())
		_logFileStream.close();
}

Logger &Logger::operator<<(const std::string &str) {
	if (_showInConsole)
		std::cout << str;

	if (_logFileStream.is_open())
		_logFileStream << str;

	return *this;
}