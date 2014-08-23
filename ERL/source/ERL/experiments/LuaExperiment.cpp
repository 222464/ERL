#include <erl/experiments/LuaExperiment.h>

#include <assert.h>

LuaExperiment* _pCurrentExperiment = nullptr;
int _lastHandle = 0;
std::unordered_map<int, std::shared_ptr<erl::Field2DCL>> _handleToField;

LuaExperiment::LuaExperiment()
: _pLuaState(nullptr), _fitness(0.0f)
{
}

LuaExperiment::~LuaExperiment() {
	if (_pLuaState != nullptr)
		lua_close(_pLuaState);
}

void LuaExperiment::create(const std::string &fileName) {
	_pLuaState = luaL_newstate();

	_experimentFileName = fileName;

	// Open libraries
	luaL_openlibs(_pLuaState);

	// Register functions
	lua_register(_pLuaState, "generatePhenotype", generatePhenotype);
	lua_register(_pLuaState, "deletePhenotype", deletePhenotype);
	lua_register(_pLuaState, "setPhenotypeInput", setPhenotypeInput);
	lua_register(_pLuaState, "stepPhenotype", stepPhenotype);
	lua_register(_pLuaState, "getPhenotypeOutput", getPhenotypeOutput);
	lua_register(_pLuaState, "setFitness", setFitness);
}

float LuaExperiment::evaluate(erl::Field2DGenes &fieldGenes, const erl::Field2DEvolverSettings* pSettings,
	const std::shared_ptr<cl::Image2D> &randomImage,
	const std::shared_ptr<cl::Program> &blurProgram,
	const std::shared_ptr<cl::Kernel> &blurKernelX,
	const std::shared_ptr<cl::Kernel> &blurKernelY,
	const std::vector<std::function<float(float)>> &activationFunctions,
	const std::vector<std::string> &activationFunctionNames,
	float minInitRec, float maxInitRec, erl::Logger &logger,
	erl::ComputeSystem &cs, std::mt19937 &generator)
{
	assert(_pLuaState != nullptr);

	_pCurrentExperiment = this;

	_pFieldGenes = &fieldGenes;
	_pSettings = pSettings;
	_randomImage = randomImage;
	_blurProgram = blurProgram;
	_blurKernelX = blurKernelX;
	_blurKernelY = blurKernelY;
	_activationFunctions = activationFunctions;
	_activationFunctionNames = activationFunctionNames;
	_minInitRec = minInitRec;
	_maxInitRec = maxInitRec;
	_pLogger = &logger;
	_pCs = &cs;
	_pGenerator = &generator;

	_lastHandle = 0;
	_handleToField.clear();
	
	int s = luaL_dofile(_pLuaState, _experimentFileName.c_str());

	if (s != 0) {
		std::cerr << "Error executing experiment \"" << _experimentFileName << "\":" << std::endl;
		std::cerr << lua_tostring(_pLuaState, -1) << std::endl;
		lua_pop(_pLuaState, 1);
	}

	return _fitness;
}

int generatePhenotype(lua_State* pLuaState) {
	int argc = lua_gettop(pLuaState);

	assert(argc == 7);

	int argWidth = lua_tonumber(pLuaState, 1);
	int argHeight = lua_tonumber(pLuaState, 2);
	int argConnectionRadius = lua_tonumber(pLuaState, 3);
	int argNumInputs = lua_tonumber(pLuaState, 4);
	int argNumOutputs = lua_tonumber(pLuaState, 5);
	int argInputRange = lua_tonumber(pLuaState, 6);
	int argOutputRange = lua_tonumber(pLuaState, 7);

	_lastHandle++;

	std::shared_ptr<erl::Field2DCL> field(new erl::Field2DCL());

	field->create(*_pCurrentExperiment->_pFieldGenes, *_pCurrentExperiment->_pCs, argWidth, argHeight, argConnectionRadius,
		argNumInputs, argNumOutputs, argInputRange, argOutputRange, _pCurrentExperiment->_randomImage,
		_pCurrentExperiment->_blurProgram, _pCurrentExperiment->_blurKernelX, _pCurrentExperiment->_blurKernelY,
		_pCurrentExperiment->_activationFunctions, _pCurrentExperiment->_activationFunctionNames,
		_pCurrentExperiment->_minInitRec, _pCurrentExperiment->_maxInitRec, *_pCurrentExperiment->_pGenerator, *_pCurrentExperiment->_pLogger);

	_handleToField[_lastHandle] = field;

	lua_pushnumber(pLuaState, _lastHandle);

	return 1; // number of return values
}

int deletePhenotype(lua_State* pLuaState) {
	int argc = lua_gettop(pLuaState);

	assert(argc == 1);

	int arg = lua_tonumber(pLuaState, 1);

	std::unordered_map<int, std::shared_ptr<erl::Field2DCL>>::iterator it = _handleToField.find(arg);

	if (it != _handleToField.end())
		_handleToField.erase(it);

	return 0;
}

int setPhenotypeInput(lua_State* pLuaState) {
	int argc = lua_gettop(pLuaState);

	assert(argc == 3);

	int argField = lua_tonumber(pLuaState, 1);
	int argIndex = lua_tonumber(pLuaState, 2);
	float argValue = lua_tonumber(pLuaState, 3);

	_handleToField[argField]->setInput(argIndex, argValue);

	return 0;
}

int stepPhenotype(lua_State* pLuaState) {
	int argc = lua_gettop(pLuaState);

	assert(argc == 3);

	int argField = lua_tonumber(pLuaState, 1);
	float argReward = lua_tonumber(pLuaState, 2);
	int argSubSteps = lua_tonumber(pLuaState, 3);

	_handleToField[argField]->update(argReward, *_pCurrentExperiment->_pCs, _pCurrentExperiment->_activationFunctions, argSubSteps, *_pCurrentExperiment->_pGenerator);

	return 0;
}

int getPhenotypeOutput(lua_State* pLuaState) {
	int argc = lua_gettop(pLuaState);

	assert(argc == 2);

	int argField = lua_tonumber(pLuaState, 1);
	int argIndex = lua_tonumber(pLuaState, 2);

	float output = _handleToField[argField]->getOutput(argIndex);

	lua_pushnumber(pLuaState, output);

	return 1;
}

int setFitness(lua_State* pLuaState) {
	int argc = lua_gettop(pLuaState);

	assert(argc == 1);

	float argFitness = lua_tonumber(pLuaState, 1);

	_pCurrentExperiment->_fitness = argFitness;

	return 0;
}