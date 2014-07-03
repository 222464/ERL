#include <erl/platform/Field2DGenesToCL.h>
#include <erl/platform/RuleToCL.h>

using namespace erl;

// Buffer is organized like so:
// nodeOutput[0..n] + typeInput + nodeRecurrentData[0..n] + c * (connectionResponse[0..n] + typeInput + connectionRecurrentData[0..n])

std::string erl::field2DGenesNodeUpdateToCL(erl::Field2DGenes &genes, const erl::Field2D &field,
	neat::NetworkPhenotype &connectionPhenotype, neat::NetworkPhenotype &nodePhenotype,
	const neat::NetworkPhenotype::RuleData &connectionRuleData, const neat::NetworkPhenotype::RuleData &nodeRuleData,
	const std::vector<std::string> &functionNames, int fieldWidth, int fieldHeight, int connectionRadius, int numInputs, int numOutputs)
{
	std::string code = "";

	// Add header
	code +=
		"/*\n"
		"ERL\n"
		"\n"
		"Generated OpenCL kernel\n"
		"*/\n"
		"\n"
		"// Samplers for input and random\n"
		"const sampler_t normalizedRepeatNearestSampler = CLK_NORMALIZED_COORDS_FALSE |\n"
		"	CLK_ADDRESS_REPEAT |\n"
		"	CLK_FILTER_NEAREST;\n"
		"\n"
		"// Dimensions of field\n"
		"const int fieldWidth = " + std::to_string(fieldWidth) + ";\n"
		"const int fieldHeight = " + std::to_string(fieldHeight) + ";\n"
		"const float fieldWidthInv = " + std::to_string(1.0f / fieldWidth) + ";\n"
		"const float fieldHeightInv = " + std::to_string(1.0f / fieldHeight) + ";\n"
		"const float numInputs = " + std::to_string(numInputs) + ";\n"
		"const float numOutputs = " + std::to_string(numOutputs) + ";\n"
		"\n"
		"// Connection offsets\n"
		"const char2 offsets[" + std::to_string(field.getNumConnections()) + "] = {\n";

	for (int x = -connectionRadius; x <= connectionRadius; x++) {
		code += "	";

		for (int y = -connectionRadius; y <= connectionRadius; y++) {
			code += "(char2)(" + std::to_string(x) + ", " + std::to_string(y) + "), ";
		}

		code += "\n";
	}

	code.pop_back();
	code.pop_back();
	code.pop_back();

	code +=
		"\n"
		"}\n"
		"\n"
		"// Connection update rule\n";

	// Generate rules for all nets
	code += ruleToCL(connectionPhenotype, connectionRuleData, "connectionRule", functionNames);

	code += "\n// Activation update rule\n";

	code += ruleToCL(nodePhenotype, nodeRuleData, "activationRule", functionNames);

	// Other constants, and kernel definition
	code +=
		"\n"
		"// Data sizes\n"
		"const int nodeAndConnectionsSize = " + std::to_string(field.getNodeAndConnectionsSize()) + ";\n"
		"const int connectionSize = " + std::to_string(field.getConnectionSize()) + ";\n"
		"const int nodeSize = " + std::to_string(field.getNodeSize()) + ";\n"
		"\n"
		"// The kernel\n"
		"void kernel nodeUpdate(global const float* source, global float* destination, read_only image2d_t typeImage, read_only image1d_t inputImage, write_only image1d_t outputImage, read_only image2d_t randomImage, float2 randomSeed, float reward) {\n"
		"	int nodeIndex = get_global_id(0);\n"
		"	int nodeStartOffset = nodeIndex * nodeAndConnectionsSize;\n"
		"	int connectionsStartOffset = nodeStartOffset + nodeSize;\n"
		"	int2 nodePosition = (int2)(nodeIndex % fieldWidth, nodeIndex / fieldHeight);\n"
		"	float2 normalizedCoords = (float2)nodePosition * float2(fieldWidthInv, fieldHeightInv);\n"
		"	float nodeType = source[nodeStartOffset + " + std::to_string(genes.getNodeOutputSize()) + "];\n"
		"\n"
		"	uchar2 nodeInputOutputIndicesPlusOne = read_imageui(typeImage, normalizedRepeatNearestSampler, nodePosition).rg;\n"
		"\n"
		"	// Update connections\n";

	// Declare response accumulators
	for (int i = 0; i < genes.getConnectionResponseSize(); i++) {
		code += "	float responseSum" + std::to_string(i) + ";\n";
	}

	// Initialize response accumulators
	code +=
		"	if (nodeInputOutputIndicesPlusOne.r == 0) {\n";

	for (int i = 0; i < genes.getConnectionResponseSize(); i++) {
		code += "	responseSum" + std::to_string(i) + " = 0;\n";
	}

	code +=
		"	}\n"
		"	else {\n";

	for (int i = 0; i < genes.getConnectionResponseSize(); i++) {
		code += "	responseSum" + std::to_string(i) + " = read_imagef(inputImage, normalizedRepeatNearestSampler, (nodeInputOutputIndicesPlusOne.r - 1) * " + std::to_string(genes.getConnectionResponseSize()) + " + " + std::to_string(i) + ");\n";
	}

	code +=
		"	}\n"
		"\n"
		"	for (int ci = 0; ci < numConnections; ci++) {\n"
		"		int2 connectionNodePosition = nodePosition + offsets[ci]\n"
		"\n"
		"		// Wrap the coordinates around\n"
		"		connectionNodePosition.x = connectionNodePosition.x % fieldWidth;\n"
		"		connectionNodePosition.y = connectionNodePosition.x % fieldHeight;\n"
		"		connectionNodePosition.x = connectionNodePosition.x < 0 ? connectionNodePosition.x + fieldWidth : connectionNodePosition.x;\n"
		"		connectionNodePosition.y = connectionNodePosition.y < 0 ? connectionNodePosition.y + fieldHeight : connectionNodePosition.y;\n"
		"\n"
		"		int connectionNodeIndex = connectionNodePosition.x + connectionNodePosition.y * fieldWidth;\n"
		"		int connectionNodeStartOffset = connectionNodeIndex * nodeAndConnectionSize;\n"
		"		int connectionStartOffset = connectionsStartOffset + ci * connectionSize;\n"
		"\n";

	// Provide temporaries for holding outputs
	for (int i = 0; i < genes.getConnectionResponseSize(); i++) {
		code += "		float response" + std::to_string(i) + ";\n";
	}

	// Assign changeable recurrent values
	for (int i = 0; i < connectionRuleData._numRecurrentSourceNodes; i++) {
		code += "		float connectionRec" + std::to_string(i) + " =  source[connectionStartOffset + " + std::to_string(genes.getConnectionResponseSize() + i) + "];\n";
	}

	code += "\n"
		"		connectionRule(";

	// Add inputs
	for (int i = 0; i < genes.getNodeOutputSize(); i++) {
		code += "source[connectionStartOffset + " + std::to_string(i) + "], ";
	}

	// Type, random, and reward inputs
	code +=
		"nodeType, read_imagef(randomImage, normalizedRepeatNearestSampler, connectionNodePosition + nodePosition).x, reward, ";

	// Add outputs
	for (int i = 0; i < genes.getConnectionResponseSize(); i++) {
		code += "&response" + std::to_string(i) + ", ";
	}

	// Add recurrent connections
	for (int i = 0; i < connectionRuleData._numRecurrentSourceNodes; i++) {
		code += "&connectionRec" + std::to_string(i) + ", ";
	}

	code.pop_back();
	code.pop_back();

	code +=
		");\n"
		"\n"
		"		// Add response to sum and assign to destination buffer\n";

	for (int i = 0; i < genes.getConnectionResponseSize(); i++) {
		code += "		destination[connectionNodeStartOffset + " + std::to_string(i) + "] = response" + std::to_string(i) + ";\n";
	}

	code +=
		"\n"
		"		// Accumulate response\n";

	for (int i = 0; i < genes.getConnectionResponseSize(); i++) {
		code += "		responseSum" + std::to_string(i) + " += response" + std::to_string(i) + ";\n";
	}

	code +=
		"\n"
		"		// Assign recurrent values to destination buffer\n";

	for (int i = 0; i < connectionRuleData._numRecurrentSourceNodes; i++) {
		code += "		destination[connectionStartOffset + " + std::to_string(genes.getConnectionResponseSize() + i) + "] = connectionRec" + std::to_string(i) + ";\n";
	}

	// ----------------------------------------------------------- Finish block -----------------------------------------------------------

	code +=
		"	}\n"
		"\n";

	// Update activation
	for (int i = 0; i < genes.getNodeOutputSize(); i++) {
		code += "	float output" + std::to_string(i) + ";\n";
	}

	// Assign changeable recurrent values
	for (int i = 0; i < nodeRuleData._numRecurrentSourceNodes; i++) {
		code += "	float nodeRec" + std::to_string(i) + " =  source[nodeStartOffset + " + std::to_string(genes.getNodeOutputSize() + 3 + i) + "];\n";
	}

	code += "\n"
		"	activationRule(";

	// Add inputs
	for (int i = 0; i < genes.getConnectionResponseSize(); i++) {
		code += "responseSum" + std::to_string(i) + ", ";
	}

	// Type, random, and reward inputs
	code +=
		"nodeType, read_imagef(randomImage, normalizedRepeatNearestSampler, nodePosition + int2(-1, -1)).x, reward, ";

	// Add outputs
	for (int i = 0; i < genes.getNodeOutputSize(); i++) {
		code += "&output" + std::to_string(i) + ", ";
	}

	// Add recurrent connections
	for (int i = 0; i < nodeRuleData._numRecurrentSourceNodes; i++) {
		code += "&nodeRec" + std::to_string(i) + ", ";
	}

	code.pop_back();
	code.pop_back();

	code +=
		");\n"
		"\n"
		"	// Assign to destination buffer\n";

	for (int i = 0; i < genes.getConnectionResponseSize(); i++) {
		code += "	destination[nodeStartOffset + " + std::to_string(i) + "] = output" + std::to_string(i) + ";\n";
	}

	code +=
		"\n"
		"	// Assign recurrent values to destination buffer\n";

	for (int i = 0; i < connectionRuleData._numRecurrentSourceNodes; i++) {
		code += "	destination[nodeStartOffset + " + std::to_string(genes.getConnectionResponseSize() + 3 + i) + "] = nodeRec" + std::to_string(i) + ";\n";
	}

	// Finish kernel by writing output if it exists
	code +=
		"\n"
		"	if (nodeInputOutputIndicesPlusOne.g != 0) {\n";

	for (int i = 0; i < genes.getNodeOutputSize(); i++) {
		code += "	write_imagef(outputImage, (nodeInputOutputIndicesPlusOne.g - 1) * " + std::to_string(genes.getNodeOutputSize()) + " + " + std::to_string(i) + ");\n";
	}

	code +=
		"	}\n"
		"}";

	return code;
}