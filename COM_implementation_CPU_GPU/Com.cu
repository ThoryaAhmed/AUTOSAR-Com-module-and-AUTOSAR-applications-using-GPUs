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
