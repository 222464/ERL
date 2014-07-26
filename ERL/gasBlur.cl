/*
ERL

Gas buffer blur kernels
*/

constant float importances[9] = {
	0.05f, 0.09f, 0.12f, 0.15f, 0.16f, 0.15f, 0.12f, 0.09f, 0.05f
};

float getSampleWrapX(global const float* source, int3 pixelPos, int width, int height, int numNodes) {
	// Wrap X only
	pixelPos.x = pixelPos.x % width;
	pixelPos.x = pixelPos.x < 0 ? pixelPos.x + width : pixelPos.x;

	return source[pixelPos.x + pixelPos.y * width + pixelPos.z * numNodes];
}

float getSampleWrapY(global const float* source, int3 pixelPos, int width, int height, int numNodes) {
	// Wrap Y only
	pixelPos.y = pixelPos.y % height;
	pixelPos.y = pixelPos.y < 0 ? pixelPos.y + height : pixelPos.y;
	
	return source[pixelPos.x + pixelPos.y * width + pixelPos.z * numNodes];
}

float getSample(global const float* source, int3 pixelPos, int width, int height, int numNodes) {
	// Wrap XY
	pixelPos.x = pixelPos.x % width;
	pixelPos.x = pixelPos.x < 0 ? pixelPos.x + width : pixelPos.x;
	pixelPos.y = pixelPos.y % height;
	pixelPos.y = pixelPos.y < 0 ? pixelPos.y + height : pixelPos.y;
	
	return source[pixelPos.x + pixelPos.y * width + pixelPos.z * numNodes];
}

void kernel blurX(global const float* source, global float* destination, int width, int height, int numNodes) {
	int3 pixelPos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float sum = 0.0f;

	for (int dx = -4; dx <= 4; dx++)
		sum += getSampleWrapX(source, pixelPos + (int3)(dx, 0, 0), width, height, numNodes) * importances[dx + 4];
 
	destination[pixelPos.x + pixelPos.y * width + pixelPos.z * numNodes] = sum;
}

void kernel blurY(global const float* source, global float* destination, int width, int height, int numNodes) {
	int3 pixelPos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float sum = 0.0f;

	for (int dy = -4; dy <= 4; dy++)
		sum += getSampleWrapY(source, pixelPos + (int3)(0, dy, 0), width, height, numNodes) * importances[dy + 4];
 
	destination[pixelPos.x + pixelPos.y * width + pixelPos.z * numNodes] = sum;
}