--[[
	ERL

	Language Modelling Task
	Based off of pole balancing experiment

	Daniel Housholder
]]--

--============HYPERPARAMETERS============--
local fieldWidth = 32
local fieldHeight = 32
local connectionRadius = 3
local trainingIterations = 1000
local testIterations = 1000
local stepSize = 16
--NEEDS TO BE CHANGED IF DIRECTORY CHANGES
local corpusFileName = "experiments/textcorpus.txt"
local stringLength = 10
--============UTILITY FUNCTIONS============-

function save(filename, data)
	local file = assert(io.open(filename, "w"))
	file:write(data)
	file:close()
end

function open(filename)
	local f = assert(io.open(filename, "r"))
    local t = f:read("*all")
    f:close()
	return t
end


-- Generate the field

--generatePhenotype(fieldWidth, fieldHeight, connectionRadius, numInputs, numOutputs, inputRange, outputRange)
local handle = generatePhenotype(fieldWidth, fieldHeight, connectionRadius, 27, 27, 1, 1) --input and output range should be 1?

corpus = open(corpusFileName)
--corpus = string.lower(corpus)
fitness = 0
stepPhenotype(handle, 0, 1)
str = ""
for i = 1, trainingIterations+testIterations do
	local character = string.sub(corpus, i, i)
	local byte = string.byte(character)
	local value
	if byte > 96 and byte < 123 then value = byte - 96
		else value = 0 end

	--if value > 26 or value < 0 then error("Value incorrect.") end
	local total = 0
	local bestProb = 0
	local best = 0
	for n = 0, 26 do
		local prob = math.max(-1, getPhenotypeOutput(handle, n)) + 1
		if prob > bestProb then bestProb = prob best = n end
		total = total + prob
	end
	if i > trainingIterations+testIterations-stringLength then str = str..string.char(best+96) end
	local reward = (math.max(-1, getPhenotypeOutput(handle, value))+1)/total
	--if not (reward < 1) or not (reward > 0) then error("reward out of range "..reward.." "..total.." value: "..value.." "..i.." "..getPhenotypeOutput(handle, value)) end
	if i > trainingIterations then
		fitness = fitness + math.log(reward)
	end

	for n = 0, 26 do
		setPhenotypeInput(handle, n, 0)
	end
	setPhenotypeInput(handle, value, 1)
	stepPhenotype(handle, reward, stepSize)
end
print(str)
setFitness(fitness)
