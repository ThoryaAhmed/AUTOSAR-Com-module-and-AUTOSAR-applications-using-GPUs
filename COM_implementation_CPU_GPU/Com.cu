/*********************************************************************************************************************************
Service name:               Com_ReceiveShadowSignal
Service ID:                    0x0f
Parameters (in):           SignalId--> Id of group signal to be received.
						   SignalDataPtr --> Reference to the group signal data in which to store the received data.
Parameters (inout):            None
Parameters (out):              None
Return value:                  None
Description:        The service Com_ReceiveShadowSignal updates the group signal which is referenced by SignalDataPtr with the data in the shadow buffer.
*******************************************************************************************************************************/
__global__ void Com_Recieve_shadow_signal(Com_SignalIdType* SignalId_arr, void* SignalDataPtr_arr, uint8* Buffer)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < ComMaxGroupSignalCnt)
	{
		if (SignalId_arr[id] < ComMaxGroupSignalCnt)
		{
			((uint8*)SignalDataPtr_arr)[id] = Buffer[SignalId_arr[id]];
		}
	}
}
void Com_ReceiveShadowSignal_GPU(Com_SignalIdType* SignalId_arr, void* SignalDataPtr_arr)
{
	Com_SignalIdType* SignalId_GPU_arr;

	uint8* SignalDataPtr_GPU_arr, * Buffer_GPU;
	uint8 Buffer[ComMaxGroupSignalCnt];
	cudaEvent_t start, stop;
	cudaMalloc((void**)&SignalId_GPU_arr, ComMaxGroupSignalCnt * sizeof(Com_SignalIdType));
	cudaMalloc((void**)&SignalDataPtr_GPU_arr, ComMaxGroupSignalCnt * sizeof(uint8));
	cudaMalloc((void**)&Buffer_GPU, ComMaxGroupSignalCnt * sizeof(uint8));

	for (int i = 0; i < ComMaxGroupSignalCnt; i++)
	{
		Buffer[i] = *(Com.ComConfig.ComGroupSignal[i].ComBufferRef);
	}

	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1);
	cudaMemcpy(SignalId_GPU_arr, SignalId_arr, ComMaxGroupSignalCnt * sizeof(Com_SignalIdType), cudaMemcpyHostToDevice);
	cudaMemcpy(Buffer_GPU, Buffer, ComMaxGroupSignalCnt * sizeof(uint8), cudaMemcpyHostToDevice);
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&gpu_time[0], start1, stop1);
	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2);
	Com_Recieve_shadow_signal << <THREADS_PER_BLOCK, BLOCKS >> > (SignalId_GPU_arr, SignalDataPtr_GPU_arr, Buffer_GPU);
	cudaDeviceSynchronize();
	cudaEventRecord(stop2);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&gpu_time[1], start2, stop2);
	cudaEvent_t start3, stop3;
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	cudaEventRecord(start3);
	cudaMemcpy(SignalDataPtr_arr, SignalDataPtr_GPU_arr, ComMaxGroupSignalCnt * sizeof(uint8), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop3);
	cudaEventSynchronize(stop3);
	cudaEventElapsedTime(&gpu_time[2], start3, stop3);
}
void Com_ReceiveShadowSignal(Com_SignalIdType SignalId, void* SignalDataPtr)
{
	uint8 ComGroupSignalIndex;

	/* Check that the group signal ID is a valid ID*/
	if (SignalId < ComMaxGroupSignalCnt)
	{
		/*Find GroupSignal with such ID*/
		for (ComGroupSignalIndex = 0; ComGroupSignalIndex < ComMaxGroupSignalCnt; ComGroupSignalIndex++)
		{
			if (Com.ComConfig.ComGroupSignal[ComGroupSignalIndex].ComHandleId == SignalId)
			{
				memcpy(SignalDataPtr, Com.ComConfig.ComGroupSignal[ComGroupSignalIndex].ComBufferRef, Com.ComConfig.ComGroupSignal[ComGroupSignalIndex].ComSignalLength);

				return;
			}
			else
			{

			}
		}
	}
	else
	{

	}
}

uint8 Com_ReceiveSignalGroup_GPU(Com_SignalGroupIdType SignalGroupId)
{
	/* Definition of Variables */
	Com_GroupSignalType* ComGroupSignalLocal = NULL;
	Com_IPduType* ComIPduLocal = NULL;

	/***************************/

	/***************************************************************************************************************************/
	/*          The service Com_ReceiveSignalGroup shall copy the received singal group from I-PDU to shadow buffer            */
	/* After this call, the group signals could be copied from the shadow buffer to the rte by calling Com_ReceiveShadowSignal */
	/***************************************************************************************************************************/

	/* Check that the id is valid */
	if (SignalGroupId <= ComMaxSignalGroupCnt)
	{
		/* Get the signal */
		const Com_SignalGroupType* ComSignalGroup = &Com.ComConfig.ComSignalGroup[SignalGroupId];

		/* Get IPDU */
		ComIPduLocal = &Com.ComConfig.ComIPdu[ComSignalGroup->ComIPduHandleId];

		uint8 ComSignalGroupIndex = ComSignalGroup->ComIPduHandleIndex;

		/* Initialize GPU Streams */
		int N_STREAMS = ComIPduLocal->ComIPduSignalGroupRef[ComSignalGroupIndex]->ComGroupSignalsNumbers;
		cudaStream_t* stream = new cudaStream_t[N_STREAMS];

		for (uint8 ComGroupSignalIndex = 0; ComGroupSignalIndex < N_STREAMS; ComGroupSignalIndex++)
		{
			/* Create Stream */
			cudaStreamCreate(&stream[ComGroupSignalIndex]);

			/*Get Group Signal*/
			ComGroupSignalLocal = ComIPduLocal->ComIPduSignalGroupRef[ComSignalGroupIndex]->ComGroupSignalRef[ComGroupSignalIndex];

			/* GPU Operations Begin here */
			uint8 threads = NUM_THREADS;
			uint16 blocks = ceil(ComGroupSignalLocal->ComBitSize / (float)threads) * Signals_Factor;
			uint8 length = ComIPduLocal->ComIPduLength;

			uint8* h_output_values = ComGroupSignalLocal->ComBufferRef;
			uint8* h_input_values = ComIPduLocal->ComBufferRef;
			uint8* h_data = h_input_values;
			uint16 h_size_out = ComGroupSignalLocal->ComBitSize;
			uint8 h_size_in = ComIPduLocal->ComIPduLength * 8;
			uint8 h_bit_position = ComGroupSignalLocal->ComBitPosition;

			uint8 h_length_in = ComIPduLocal->ComIPduLength;
			uint8 h_length_out = ComGroupSignalLocal->ComSignalLength;
			
			// declare GPU memory pointers
			uint8* d_output_values, * d_input_values , *d_data;
			uint16* d_size_out, * d_size_in;
			uint32* d_bit_position;

			cudaEvent_t start3, stop3;
			cudaEventCreate(&start3);
			cudaEventCreate(&stop3);
			cudaEventRecord(start3);

			// allocate GPU memory
			cudaMalloc((void**)&d_output_values, h_length_out * sizeof(uint8));
			cudaMalloc((void**)&d_input_values, h_length_in * sizeof(uint8));
			cudaMalloc((void**)&d_data, Signals_Factor* h_length_in * sizeof(uint8));

			// transfer the input array to the GPU
			cudaMemcpyAsync(d_output_values, h_output_values, h_length_out * sizeof(uint8), cudaMemcpyHostToDevice);
			cudaMemcpyAsync(d_input_values, h_input_values, h_length_in * sizeof(uint8), cudaMemcpyHostToDevice);
			if (Signals_Factor < 1000) {
				cudaMemcpyAsync(d_data, h_input_values, Signals_Factor * h_length_in, cudaMemcpyHostToDevice);
				cudaMemcpyAsync(d_data, h_input_values, Signals_Factor * h_length_out, cudaMemcpyHostToDevice);
			}
			else {
				cudaMemcpyAsync(d_data, h_input_values, Signals_Factor/2 * h_length_in, cudaMemcpyHostToDevice);
				cudaMemcpyAsync(d_data, h_input_values, Signals_Factor/2 * h_length_out, cudaMemcpyHostToDevice);
				cudaMemcpyAsync(d_data, h_input_values, Signals_Factor/2 * h_length_out, cudaMemcpyHostToDevice);
				cudaMemcpyAsync(d_data, h_input_values, Signals_Factor/2 * h_length_out, cudaMemcpyHostToDevice);
			}

			cudaEventRecord(stop3);
			cudaEventSynchronize(stop3);


			cudaEventElapsedTime(&gpu_time[0], start3, stop3);

			cudaEvent_t start2, stop2;
			cudaEventCreate(&start2);
			cudaEventCreate(&stop2);
			cudaEventRecord(start2);
			///////////////////////////////////////////////* Output Values *//* Input Values *//*size o/p*//*size of i/p*//* bit position of The signal */
			Unpacking_Bits_kernel << < blocks, threads, threads * sizeof(uint8), stream[ComGroupSignalIndex] >> > (d_output_values, d_input_values, h_size_out, h_size_in, h_bit_position);
			cudaDeviceSynchronize();

			cudaEventRecord(stop2);
			cudaEventSynchronize(stop2);


			cudaEventElapsedTime(&gpu_time[1], start2, stop2);
			cudaEventDestroy(start2);
			cudaEventDestroy(stop2);
			// Return the results to the signal

			cudaEvent_t start4, stop4;
			cudaEventCreate(&start4);
			cudaEventCreate(&stop4);
			cudaEventRecord(start4);

			// transfer the input array to the GPU
			cudaMemcpyAsync(h_output_values, d_output_values,  h_length_out * sizeof(uint8), cudaMemcpyDeviceToHost);
			if (Signals_Factor < 1000) {
				cudaMemcpyAsync(h_data, d_data, Signals_Factor * h_length_out * sizeof(uint8), cudaMemcpyDeviceToHost);
			}
			else {
				cudaMemcpyAsync(h_data, d_data, Signals_Factor/2 * h_length_out * sizeof(uint8), cudaMemcpyDeviceToHost);
				cudaMemcpyAsync(h_data, d_data, Signals_Factor/2 * h_length_out * sizeof(uint8), cudaMemcpyDeviceToHost);

			}
			cudaEventRecord(stop4);
			cudaEventSynchronize(stop4);

			cudaEventElapsedTime(&gpu_time[2], start4, stop4);

			cudaFree(d_data);
		}


		//}

		//cudaDeviceReset();
		free(stream);
		return E_OK;

	}
	else
	{
	}

	return COM_SERVICE_NOT_AVAILABLE;
}
