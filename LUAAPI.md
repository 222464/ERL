#Lua API

## Tutorial

To get started with the Lua experiment scripting API, create a new .lua file. To make the ERL host program aware of your experiment, you must add the path to the .lua file into experiments.txt.
Once you have done this, ERL should detect your experiment when your run it in training mode.

An experiment takes an ERL genotype as an input and gives a fitness value as an output.

There are 6 functions that are part of the Lua API:
* `handle generatePhenotype(fieldWidth, fieldHeight, connectionRadius, numInputs, numOutputs, inputRange, outputRange)` - creates a new phenotype from the genotype associated with this experiment. Returns a handle to the phenotype.
* `deletePhenotype(handle)` - deletes a phenotype previously created with generatePhenotype
* `setPhenotypeInput(handle, index, value)` - sets the input to a field denoted by handle to the specified value
* `stepPhenotype(handle, reward, substeps)` - steps (simulates) the phenotype specified by handle with given reward and a number of substeps (simulation steps)
* `getPhenotypeOutput(handle, index)` - gets the output of the phenotype denoted by handle at the specified index
* `setFitness(value)` - sets the fitness for this experiment. This function must be called at least once per experiment!

Experiments typically follow this pattern of API calls:

`
h = generatePhenotype(...)

for (number of simulation steps) do
	<read sensors (environment)>

	setPhenotypeInput(h, 0, sensorValue0)
	setPhenotypeInput(h, 1, sensorValue1)
	...
	setPhenotypeInput(h, n, sensorValueN)
	
	stepPhenotype(h, r, 24)
	
	output0 = getPhenotypeOutput(h, 0)
	output1 = getPhenotypeOutput(h, 1)
	...
	outputN = getPhenotypeOutput(h, n)
	
	<perform action>
end

setFitness(myAwesomelyHighFitness)
`

Note that setFitness must be called at least once, since it tells the ERL host program how well the genotype performed in this experiment!

Happy experimenting!
