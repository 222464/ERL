/*
ERL

Buffer adaption kernel
*/

void kernel adapt(global const float* source, write_only image2d_t destination, int width, int nodeAndConnectionsSize, int nodeOutputSize) {
	int2 pixelPos = (int2)(get_global_id(0), get_global_id(1));
	int lPos = pixelPos.x + pixelPos.y * width;

	float color0 = source[lPos * nodeAndConnectionsSize + 0];

	write_imagef(destination, pixelPos, (float4)(0.5, 0.2, 0.8, 1.0));
}