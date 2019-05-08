//FIXME !!!! radius must be < size[dir] !!!!

//FIXME: x kernel does non-coalesced reads/writes
__global__ void filterXKernel(
	uint radius, const float* __restrict__ pData, float* __restrict__ pOut,
	uint sizeX, uint sizeY, uint sizeZ, uint overlap2,
	const float* __restrict__ pLeft, uint sizeLeftX, const float* __restrict__ pRight, uint sizeRightX
)
{
	uint y = blockIdx.x * blockDim.x + threadIdx.x;
	uint z = blockIdx.y * blockDim.y + threadIdx.y;

	if(y >= sizeY || z >= sizeZ) return;

	pData  += sizeX      * (y + sizeY * z);
	pOut   += sizeX      * (y + sizeY * z);
	pLeft  += sizeLeftX  * (y + sizeY * z);
	pRight += sizeRightX * (y + sizeY * z);


	float sum = 0.0f;

	// get initial values from left
	for(uint x = max(sizeLeftX, overlap2 + radius) - overlap2 - radius; x < max(sizeLeftX, overlap2) - overlap2; x++) {
		sum += pLeft[x];
	}

	// get initial values from center
	for(uint x = 0; x < min(radius + 1, sizeX); x++) {
		sum += pData[x];
	}


	float scale = 1.0f / float(2 * radius + 1);


	uint x = 0;

	pOut[x++] = sum * scale;


	// input from left and center
	while(x < min(radius + 1, max(sizeX, radius) - radius)) {
		if(sizeLeftX + x >= overlap2 + radius + 1)
			sum -= pLeft[sizeLeftX + x - overlap2 - radius - 1];
		sum += pData[x + radius];

		pOut[x++] = sum * scale;
	}

	if(sizeX >= 2 * radius + 1) {
		// input from center only
		while(x < sizeX - radius) {
			sum -= pData[x - radius - 1];
			sum += pData[x + radius];

			pOut[x++] = sum * scale;
		}
	} else {
		// input from left and right
		while(x < min(radius + 1, sizeX)) {
			if(sizeLeftX + x >= overlap2 + radius + 1)
				sum -= pLeft[sizeLeftX + x - overlap2 - radius - 1];
			if(overlap2 + x + radius < sizeX + sizeRightX)
				sum += pRight[overlap2 + x + radius - sizeX];

			pOut[x++] = sum * scale;
		}
	}

	// input from center and right
	while(x < sizeX) {
		sum -= pData[x - radius - 1];
		if(overlap2 + x + radius < sizeX + sizeRightX)
			sum += pRight[overlap2 + x + radius - sizeX];

		pOut[x++] = sum * scale;
	}
}

__global__ void filterYKernel(
	uint radius, const float* __restrict__ pData, float* __restrict__ pOut,
	uint sizeX, uint sizeY, uint sizeZ, uint overlap2,
	const float* __restrict__ pLeft, uint sizeLeftY, const float* __restrict__ pRight, uint sizeRightY
)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint z = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= sizeX || z >= sizeZ) return;

	pData  += x + sizeX * sizeY      * z;
	pOut   += x + sizeX * sizeY      * z;
	pLeft  += x + sizeX * sizeLeftY  * z;
	pRight += x + sizeX * sizeRightY * z;


	float sum = 0.0f;

	// get initial values from left
	for(uint y = max(sizeLeftY, overlap2 + radius) - overlap2 - radius; y < max(sizeLeftY, overlap2) - overlap2; y++) {
		sum += pLeft[y * sizeX];
	}

	// get initial values from center
	for(uint y = 0; y < min(radius + 1, sizeY); y++) {
		sum += pData[y * sizeX];
	}


	float scale = 1.0f / float(2 * radius + 1);


	uint y = 0;

	pOut[y++] = sum * scale;


	// input from left and center
	while(y < min(radius + 1, max(sizeY, radius) - radius)) {
		if(sizeLeftY + y >= overlap2 + radius + 1)
			sum -= pLeft[(sizeLeftY + y - overlap2 - radius - 1) * sizeX];
		sum += pData[(y + radius) * sizeX];

		pOut[y++ * sizeX] = sum * scale;
	}

	if(sizeY >= 2 * radius + 1) {
		// input from center only
		while(y < sizeY - radius) {
			sum -= pData[(y - radius - 1) * sizeX];
			sum += pData[(y + radius) * sizeX];

			pOut[y++ * sizeX] = sum * scale;
		}
	} else {
		// input from left and right
		while(y < min(radius + 1, sizeY)) {
			if(sizeLeftY + y >= overlap2 + radius + 1)
				sum -= pLeft[(sizeLeftY + y - overlap2 - radius - 1) * sizeX];
			if(overlap2 + y + radius < sizeY + sizeRightY)
				sum += pRight[(overlap2 + y + radius - sizeY) * sizeX];

			pOut[y++ * sizeX] = sum * scale;
		}
	}

	// input from center and right
	while(y < sizeY) {
		sum -= pData[(y - radius - 1) * sizeX];
		if(overlap2 + y + radius < sizeY + sizeRightY)
			sum += pRight[(overlap2 + y + radius - sizeY) * sizeX];

		pOut[y++ * sizeX] = sum * scale;
	}
}

__global__ void filterZKernel(
	uint radius, const float* __restrict__ pData, float* __restrict__ pOut,
	uint sizeX, uint sizeY, uint sizeZ, uint overlap2,
	const float* __restrict__ pLeft, uint sizeLeftZ, const float* __restrict__ pRight, uint sizeRightZ
)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= sizeX || y >= sizeY) return;

	pData  += x + sizeX * y;
	pOut   += x + sizeX * y;
	pLeft  += x + sizeX * y;
	pRight += x + sizeX * y;


	uint pitch = sizeX * sizeY;


	float sum = 0.0f;

	// get initial values from left
	for(uint z = max(sizeLeftZ, overlap2 + radius) - overlap2 - radius; z < max(sizeLeftZ, overlap2) - overlap2; z++) {
		sum += pLeft[z * pitch];
	}

	// get initial values from center
	for(uint z = 0; z < min(radius + 1, sizeZ); z++) {
		sum += pData[z * pitch];
	}


	float scale = 1.0f / float(2 * radius + 1);


	uint z = 0;

	pOut[z++] = sum * scale;


	// input from left and center
	while(z < min(radius + 1, max(sizeZ, radius) - radius)) {
		if(sizeLeftZ + z >= overlap2 + radius + 1)
			sum -= pLeft[(sizeLeftZ + z - overlap2 - radius - 1) * pitch];
		sum += pData[(z + radius) * pitch];

		pOut[z++ * pitch] = sum * scale;
	}

	if(sizeZ >= 2 * radius + 1) {
		// input from center only
		while(z < sizeZ - radius) {
			sum -= pData[(z - radius - 1) * pitch];
			sum += pData[(z + radius) * pitch];

			pOut[z++ * pitch] = sum * scale;
		}
	} else {
		// input from left and right
		while(z < min(radius + 1, sizeZ)) {
			if(sizeLeftZ + z >= overlap2 + radius + 1)
				sum -= pLeft[(sizeLeftZ + z - overlap2 - radius - 1) * pitch];
			if(overlap2 + z + radius < sizeZ + sizeRightZ)
				sum += pRight[(overlap2 + z + radius - sizeZ) * pitch];

			pOut[z++ * pitch] = sum * scale;
		}
	}

	// input from center and right
	while(z < sizeZ) {
		sum -= pData[(z - radius - 1) * pitch];
		if(overlap2 + z + radius < sizeZ + sizeRightZ)
			sum += pRight[(overlap2 + z + radius - sizeZ) * pitch];

		pOut[z++ * pitch] = sum * scale;
	}
}
