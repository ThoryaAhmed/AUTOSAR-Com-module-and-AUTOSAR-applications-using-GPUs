#include "cuda_code.cuh"
#include "inc/Com.h"
#include "inc/ComTeam_Types.h"
#include "Std_Types.h"
#include "stdio.h"
#include "Gen/Com_Cfg.h"
#include "inc/ComStack_Cfg.h"


#define NUM_THREADS 8

extern Com_Type Com;


#define NUM_THREADS 8


__global__ void Unpacking_Bits_kernel(uint8* output_buffer, uint8* input_buffer, uint16 output_size, uint16 input_size, uint32 bit_position)
{
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


uint8 Com_ReceiveSignal_GPU(Com_SignalGroupIdType SignalId, void* SignalDataPtr)
{
	/* Definition of Variables */
	Com_SignalType* ComSignalLocal = NULL;
	Com_IPduType* ComIPduLocal = NULL;
	uint8 ComIPduIndex, ComSignalIndex, BitIndex;
	/***************************/

	/***************************************************************************************************************************/
	/*          The service Com_ReceiveSignalGroup shall copy the received singal group from I-PDU to shadow buffer            */
	/* After this call, the group signals could be copied from the shadow buffer to the rte by calling Com_ReceiveShadowSignal */
	/***************************************************************************************************************************/

	/* Check that the id is valid */
	if (SignalId <= ComMaxSignalCnt)
	{
		/* Find the IPdu which contains this signal group */

		/* Get the signal */
		const Com_SignalType Signal = Com.ComConfig.ComSignal[SignalId];

		/* Get the index in pdu */
		int index_in_PDU = Signal.ComHandleId % 7;

		/* Get IPDU */
		if (SignalId < 7 )
			ComIPduLocal = &Com.ComConfig.ComIPdu[0];
		else
			ComIPduLocal = &Com.ComConfig.ComIPdu[1];

		///* Initialize GPU Streams */
		//int N_STREAMS = ComIpduLocal->ComIPduSignalGroupRef[index_in_PDU]->ComGroupSignalsNumbers;
		//cudaStream_t* stream = new cudaStream_t[N_STREAMS];


		//for (int ComGroupSignalIndex = 0; ComGroupSignalIndex < N_STREAMS; ComGroupSignalIndex++)
		//{
		//	/* Create Stream */
		//	cudaStreamCreate(&stream[ComGroupSignalIndex]);

		/*Get Signal*/
		ComSignalLocal = ComIPduLocal->ComIPduSignalRef[index_in_PDU];


		/* GPU Operations Begin here */
		uint8 threads = NUM_THREADS;
		uint16 blocks = ceil(ComSignalLocal->ComBitSize / (float)threads);
		uint8 length = ComIPduLocal->ComIPduLength;

		uint8* h_output_values = ComSignalLocal->ComBufferRef;
		uint8* h_input_values = ComIPduLocal->ComBufferRef;
		uint16 h_size_out = ComSignalLocal->ComBitSize;
		uint8 h_size_in = ComIPduLocal->ComIPduLength * 8;
		uint8 h_bit_position = ComSignalLocal->ComBitPosition;

		uint8 h_length_in = ComIPduLocal->ComIPduLength;
		uint8 h_length_out = ComSignalLocal->ComSignalLength;

		int valeo = h_input_values[0];
		int ejad = h_input_values[1];

		// declare GPU memory pointers
		uint8* d_output_values, * d_input_values;
		uint16* d_size_out, * d_size_in;
		uint32* d_bit_position;

		// allocate GPU memory
		cudaMalloc((void**)&d_output_values, h_length_out * sizeof(uint8));
		cudaMalloc((void**)&d_input_values, h_length_in * sizeof(uint8));
		//cudaMalloc((void**)&d_size_out, sizeof(uint16));
		//cudaMalloc((void**)&d_size_in, sizeof(uint16));
		//cudaMalloc((void**)&d_bit_position, sizeof(uint32));

		// transfer the input array to the GPU
		cudaMemcpy(d_output_values, h_output_values, h_length_out * sizeof(uint8), cudaMemcpyHostToDevice);
		cudaMemcpy(d_input_values, h_input_values, h_length_in * sizeof(uint8), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_size_out, &h_size_out, sizeof(uint16), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_size_in, &h_size_in, sizeof(uint16), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_bit_position, &h_bit_position, sizeof(uint32), cudaMemcpyHostToDevice);


		//////////////////////////////////////////////////////// ////////////////////////* Output Values */    /* Input Values */                    /*size o/p*/         /*size of i/p*/            /* Index of The signal */
		//Unpacking_Bits_kernel << < blocks, threads >> > (d_output_values, d_input_values, h_size_out, h_size_in, h_bit_position);
		Unpacking_Bits_kernel << < blocks, threads >> > (d_output_values, d_input_values, h_size_out, h_size_in, h_bit_position);
		cudaDeviceSynchronize();

		// Return the results to the shadow buffer 
		cudaMemcpyAsync(h_output_values, d_output_values, h_length_out * sizeof(uint8), cudaMemcpyDeviceToHost);

		memcpy(SignalDataPtr, ComSignalLocal->ComBufferRef, ComSignalLocal->ComSignalLength);


		//}

		cudaDeviceReset();

		return E_OK;

	}
	else
	{
	}

	return COM_SERVICE_NOT_AVAILABLE;
}
