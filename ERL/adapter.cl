/*
ERL

Buffer adaption kernel
*/

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

void kernel adapt(global const float* source, write_only image2d_t destination, int width, int nodeAndConnectionsSize, int nodeOutputSize) {
	int2 pixelPos = (int2)(get_global_id(0), get_global_id(1));
	int lPos = pixelPos.x + pixelPos.y * width;

	float color0 = sigmoid(source[lPos * nodeAndConnectionsSize + 0] * 3.0f);

	write_imagef(destination, pixelPos, (float4)(color0, color0, color0, 1.0f));
}