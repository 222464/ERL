/*
ERL

Lua Experiment
*/

#pragma once

#include <lua.hpp>
#include <lualib.h>

#include <erl/simulation/Experiment.h>

#include <unordered_map>

class LuaExperiment : public erl::Experiment {
private:
	lua_State* _pLuaState;

	std::string _experimentFileName;

public:
	erl::Field2DGenes* _pFieldGenes;
	const erl::Field2DEvolverSettings* _pSettings;
	std::shared_ptr<cl::Image2D> _randomImage;
	std::shared_ptr<cl::Program> _blurProgram;
	std::shared_ptr<cl::Kernel> _blurKernelX;
	std::shared_ptr<cl::Kernel> _blurKernelY;
	std::vector<std::function<float(float)>> _activationFunctions;
	std::vector<std::string> _activationFunctionNames;
	float _minInitRec;
	float _maxInitRec;
	erl::Logger* _pLogger;
	erl::ComputeSystem* _pCs;
	std::mt19937* _pGenerator;

	float _fitness;

	LuaExperiment();
	~LuaExperiment();

	void create(const std::string &fileName);

	// Inherited from Experiment
	float evaluate(erl::Field2DGenes &fieldGenes, const erl::Field2DEvolverSettings* pSettings,
		const std::shared_ptr<cl::Image2D> &randomImage,
		const std::shared_ptr<cl::Program> &blurProgram,
		const std::shared_ptr<cl::Kernel> &blurKernelX,
		const std::shared_ptr<cl::Kernel> &blurKernelY,
		const std::vector<std::function<float(float)>> &activationFunctions,
		const std::vector<std::string> &activationFunctionNames,
		float minInitRec, float maxInitRec, erl::Logger &logger,
		erl::ComputeSystem &cs, std::mt19937 &generator);
};

extern LuaExperiment* _pCurrentExperiment;
extern int _lastHandle;
extern std::unordered_map<int, std::shared_ptr<erl::Field2DCL>> _handleToField;

int generatePhenotype(lua_State* pLuaState);
int deletePhenotype(lua_State* pLuaState);

int setPhenotypeInput(lua_State* pLuaState);
int stepPhenotype(lua_State* pLuaState);
int getPhenotypeOutput(lua_State* pLuaState);

int setFitness(lua_State* pLuaState);