
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// #include "./inc/Com_Types.cuh"
#include "Gen/Com_Cfg.cu"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define Grid_Width 3
#define Block_Width 3

#define BLOCKS 16
#define THREADS_PER_BLOCK 512
#define SIZE 2

#define NUM_THREADS 8

#define WIDTH 1280
#define HEIGHT 720
#define CHANNELS 3


Com_Type Com_CPU[1];
static ComState_Type ComState = COM_UNINIT;

__global__ void Unpacking_Bits_kernel();
__global__ void Packing();

__global__ void copyToSig();
__global__ void copyToIpdu();

void Com_Init(const Com_ConfigType* config);
uint8 Com_SendSignal_GPU_concept(Com_SignalIdType SignalId, const void* SignalDataPtr);
uint8 Com_ReceiveSignal_GPU_concept(Com_SignalIdType SignalId, void* SignalDataPtr);

uint8 Com_SendSignal_GPU(Com_SignalIdType SignalId, const void* SignalDataPtr);
uint8 Com_ReceiveSignal_GPU(Com_SignalIdType SignalId, void* SignalDataPtr);

uint8* d_image; uint8* d_image_out;
uint8* h_image; uint8* h_image_out;
size_t h_image_bytes = WIDTH * HEIGHT * CHANNEL_NUM * sizeof(uint8);

int main()
{
	// com init flag
	// init copies from gpu to cpu
	Com_Init(&Com_CPU->ComConfig);

	int width, height, channels;
	h_image = stbi_load("image.jpg", &width, &height, &channels, 0);

	printf("%d %d %d\n", width, height, channels);

	h_image_out = (uint8*)malloc(WIDTH * HEIGHT * CHANNEL_NUM);
	size_t h_image_out_bytes = h_image_bytes;

	cudaMalloc((void**)&d_image, h_image_bytes);
	cudaMalloc((void**)&d_image_out, h_image_out_bytes);


	// receive sigal from the cpu to gpu
	// Com_ReceiveSignal_GPU(0, d_image);

	// send signal to the cpu for router transmission
	cudaMemcpy(d_image, h_image, h_image_bytes, cudaMemcpyHostToDevice);
	// cudaMemcpy(h_image_out, d_image, h_image_bytes, cudaMemcpyDeviceToHost);
	Com_SendSignal_GPU(0, d_image);



	// h_image_out acts as the L-PDU
	stbi_write_jpg("out.jpg", width, height, CHANNEL_NUM, h_image_out, 100);


	stbi_image_free(h_image);
	free(h_image_out);

	cudaFree(d_image);
	cudaFree(d_image_out);

	return 0;
}

// copy from device to host
__global__ void copyToSig() {
	uint8* ipdu_ptr = (uint8*)Com_GPU[0].ComConfig.ComIPdu->ComBufferRef;
	uint8* sig_ptr = (uint8*)Com_GPU[0].ComConfig.ComSignal->ComBufferRef;

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int index = row * WIDTH + col;
	int channels = 3;

	if (row < HEIGHT && col < WIDTH) {
		sig_ptr[index * channels + 0] = ipdu_ptr[index * channels + 0];
		sig_ptr[index * channels + 1] = ipdu_ptr[index * channels + 1];
		sig_ptr[index * channels + 2] = ipdu_ptr[index * channels + 2];
	}
}

__global__ void copyToIpdu() {
	uint8* ipdu_ptr = (uint8*)Com_GPU[0].ComConfig.ComIPdu->ComBufferRef;
	uint8* sig_ptr = (uint8*)Com_GPU[0].ComConfig.ComSignal->ComBufferRef;

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int index = row * WIDTH + col;
	int channels = 3;

	if (row < HEIGHT && col < WIDTH) {
		ipdu_ptr[index * channels + 0] = sig_ptr[index * channels + 0];
		ipdu_ptr[index * channels + 1] = sig_ptr[index * channels + 1];
		ipdu_ptr[index * channels + 2] = sig_ptr[index * channels + 2];
	}
}

void Com_Init(const Com_ConfigType* config) {
	// get the com from CPU to GPU
	// need to assign GPU addresses but will be left till needed
	cudaMemcpyFromSymbol(Com_CPU, Com_GPU, sizeof(Com_Type));
	ComState = COM_READY;
}

uint8 Com_SendSignal_GPU_concept(Com_SignalIdType SignalId, const void* SignalDataPtr) {
	// move from gpu pointer to signal buffer
	// move from signal buffer to IPDU buffer
	// move from IPDU to cpu
	const void* signal_buf = Com_CPU[0].ComConfig.ComSignal->ComBufferRef;
	const void* IPDU_buf = Com_CPU[0].ComConfig.ComIPdu->ComBufferRef;
	size_t sizeInBytes = Com_CPU[0].ComConfig.ComSignal->ComSignalLength;

	// copy signal to signal buffer (part of AUTOSAR)
	cudaMemcpy((uint8*)signal_buf, SignalDataPtr, sizeInBytes, cudaMemcpyDeviceToDevice);

	// IPDU bit packing
	cudaMemcpy((uint8*)IPDU_buf, signal_buf, sizeInBytes, cudaMemcpyDeviceToDevice);

	// set update bit

	// move to router
	cudaMemcpy(h_image_out, IPDU_buf, sizeInBytes, cudaMemcpyDeviceToHost);
	return 0;
}

uint8 Com_ReceiveSignal_GPU_concept(Com_SignalIdType SignalId, void* SignalDataPtr) {
	// the exact oppisite of send concept
	// from router to IPDU buffer
	// copy from IPDU to signal buffer
	// copy from signal buffer to data pointer

	const void* signal_buf = Com_CPU[0].ComConfig.ComSignal->ComBufferRef;
	const void* IPDU_buf = Com_CPU[0].ComConfig.ComIPdu->ComBufferRef;
	size_t sizeInBytes = Com_CPU[0].ComConfig.ComSignal->ComSignalLength;

	cudaMemcpy((uint8*)IPDU_buf, h_image, sizeInBytes, cudaMemcpyHostToDevice);

	cudaMemcpy((uint8*)signal_buf, IPDU_buf, sizeInBytes, cudaMemcpyDeviceToDevice);

	cudaMemcpy(SignalDataPtr, signal_buf, sizeInBytes, cudaMemcpyDeviceToDevice);

	return 0;
}

uint8 Com_SendSignal_GPU(Com_SignalIdType SignalId, const void* SignalDataPtr) {
	// assuming copy signal is true


	// move from gpu pointer to signal buffer
	// move from signal buffer to IPDU buffer
	// move from IPDU to cpu

	const void* signal_buf = Com_CPU[0].ComConfig.ComSignal->ComBufferRef;
	const void* IPDU_buf = Com_CPU[0].ComConfig.ComIPdu->ComBufferRef;
	size_t sizeInBytes = Com_CPU[0].ComConfig.ComSignal->ComSignalLength;

	uint8* ComSignalLocal = (uint8*)signal_buf;
	uint8* ComIPduLocal = (uint8*)IPDU_buf;
	uint8 ComBitSize = Com_CPU[0].ComConfig.ComSignal->ComBitSize;
	uint8 ComBitPosition = Com_CPU[0].ComConfig.ComSignal->ComBitPosition;


	// copy signal to signal buffer (part of AUTOSAR)
	cudaMemcpy(ComSignalLocal, SignalDataPtr, sizeInBytes, cudaMemcpyDeviceToDevice);

	// IPDU bit packing
	//signal and ipdu index should be passed later for more complex behaviour
	//Packing << <ceil((ComBitSize) / (float)8), 8 >> >();

	dim3 dimGrid(ceil(WIDTH / 32.0), ceil(HEIGHT / 32.0), 1); dim3 dimBlock(32, 32, 1);
	copyToIpdu << <dimGrid, dimBlock >> > ();
	// set update bit

	// move to router
	cudaMemcpy(h_image_out, IPDU_buf, sizeInBytes, cudaMemcpyDeviceToHost);
	return 0;
}

uint8 Com_ReceiveSignal_GPU(Com_SignalIdType SignalId, void* SignalDataPtr) {
	// the exact oppisite of send concept
	// from router to IPDU buffer
	// copy from IPDU to signal buffer
	// copy from signal buffer to data pointer

	const void* signal_buf = Com_CPU[0].ComConfig.ComSignal->ComBufferRef;
	const void* IPDU_buf = Com_CPU[0].ComConfig.ComIPdu->ComBufferRef;
	size_t sizeInBytes = Com_CPU[0].ComConfig.ComSignal->ComSignalLength;

	uint8* ComSignalLocal = (uint8*)signal_buf;
	uint8* ComIPduLocal = (uint8*)IPDU_buf;
	uint8 ComBitSize = Com_CPU[0].ComConfig.ComSignal->ComBitSize;
	uint8 ComBitPosition = Com_CPU[0].ComConfig.ComSignal->ComBitPosition;
	uint8 ComIPduLengthBits = Com_CPU[0].ComConfig.ComIPdu->ComIPduLength * 8;

	// copy LPDU to IPDU
	cudaMemcpy((uint8*)IPDU_buf, h_image, sizeInBytes, cudaMemcpyHostToDevice);

	// unpacking
	Unpacking_Bits_kernel << <ceil((ComBitSize) / (float)8), 8 >> > ();


	cudaMemcpy(SignalDataPtr, signal_buf, sizeInBytes, cudaMemcpyDeviceToDevice);
	// cudaMemcpy(h_image_out, ComSignalLocal, sizeInBytes, cudaMemcpyDeviceToHost);

	return 0;
}








__global__ void Packing()
{
	uint8* ComIPduLocal_GPU = (uint8*)Com_GPU[0].ComConfig.ComIPdu->ComBufferRef;
	uint8* ComSignalLocal_GPU = (uint8*)Com_GPU[0].ComConfig.ComSignal->ComBufferRef;
	uint8 ComBitSize_GPU = Com_GPU[0].ComConfig.ComSignal->ComBitSize;
	uint8 ComBitPosition_GPU = Com_GPU[0].ComConfig.ComSignal->ComBitPosition;

	uint32 Index = threadIdx.x + blockIdx.x * blockDim.x;
	if (Index >= 0 && Index < (ComBitSize_GPU))
	{
		if (ComSignalLocal_GPU[Index / 8] >> (Index % 8) & 1)
		{
			atomicOr((int*)&ComIPduLocal_GPU[(Index + ComBitPosition_GPU) / 8], (int)1 << ((Index + ComBitPosition_GPU) % 8));
			// ComIPduLocal_GPU[(Index + ComBitPosition_GPU) / 8] |= 1 << ((Index + ComBitPosition_GPU)%8);
		}
		else
		{
			atomicAnd((int*)&ComIPduLocal_GPU[(Index + ComBitPosition_GPU) / 8], (int)~(1 << ((Index + ComBitPosition_GPU) % 8)));
			// ComIPduLocal_GPU[(Index + ComBitPosition_GPU) / 8] &= ~(1 << ((Index + ComBitPosition_GPU)%8));
		}
		// ComIPduLocal_GPU[ComUpdateBitPosition_GPU / 8] |= 1 << (ComUpdateBitPosition_GPU%8);
		//*ComTeamSignalUpdated_GPU = true;
	}
}


__global__ void Unpacking_Bits_kernel()
{
	uint8* output_buffer = (uint8*)Com_GPU[0].ComConfig.ComSignal->ComBufferRef;
	uint8* input_buffer = (uint8*)Com_GPU[0].ComConfig.ComIPdu->ComBufferRef;
	uint8 output_size = Com_GPU[0].ComConfig.ComSignal->ComBitSize;
	uint8 input_size = Com_GPU[0].ComConfig.ComIPdu->ComIPduLength * 8;
	uint8 bit_position = Com_GPU[0].ComConfig.ComSignal->ComBitPosition;

	const uint16 index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint16 tid = threadIdx.x;

	extern __shared__ uint8 sdata[8];
	int valeo = input_buffer[1];
	if (index >= input_size)
	{
		return;
	}

	if (index + ((bit_position + index) / 8) * 8 >= bit_position && index + ((bit_position + index) / 8) * 8 < bit_position + output_size) {

		//tid -> thread index + No of threads = 8
		sdata[tid] = (input_buffer[(bit_position + index) / 8] >> (tid) % 8) & 1;
		sdata[tid] = sdata[tid] << (index % 8);
	}
	else
	{
		sdata[tid] = 0;
	}

	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s) {

			sdata[tid] |= sdata[tid + s];

			// check all threads end the current phase
			__syncthreads();
			// go to next phase
		}

	}


	if (tid == 0)
	{
		output_buffer[blockIdx.x] = sdata[0];
	}
}

