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

/*********************************************************************************************************************************
 Service name:               Com_UpdateShadowSignal_GPU
 Service ID:                    0x0c
 Parameters (in):           SignalId--> Id of group signal to be updated.(unit16)
							SignalDataPtr --> Reference to the group signal data to be updated.
 Parameters (inout):            None
 Parameters (out):              None
 Return value:                  None
 Description:        The service Com_UpdateShadowSignal updates a group signal with the data referenced by SignalDataPtr.
 *******************************************************************************************************************************/
__global__ void copy_data_to_shadowBuffer_on_Kernel(uint8* d_signalgroup_in, uint8 SignalDataPtr, uint8 ID)
{
	uint8 ThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;

	/* Check that the group signal ID is a valid ID*/

	if (ThreadIndex < ComMaxGroupSignalCnt && ThreadIndex == ID)
	{
		/* Copy group signal to group signal buffer */
		d_signalgroup_in[0] = SignalDataPtr;
	}


}
void Com_UpdateShadowSignal_GPU(Com_SignalIdType SignalId, const void* SignalDataPtr)
{
	uint8 SignalData = *((uint8*)SignalDataPtr);
	Com_GroupSignalType ComGroupSignalLocal = Com.ComConfig.ComGroupSignal[SignalId];
	/* Get the group_signal ID */
	uint8 ID = Com.ComConfig.ComGroupSignal[SignalId].ComHandleId;

	uint8* h_data = 0;

	uint8 h_Length = ComGroupSignalLocal.ComSignalLength;
	/*GPU Variables*/
	uint8* d_GroupSignalLocal;
	uint8* d_data;

	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1);

	/*allocate GPU memory*/
	cudaMalloc((void**)&d_GroupSignalLocal, h_Length * sizeof(uint8));
	cudaMalloc((void**)&d_data, Signals_Factor * h_Length * sizeof(uint8));

	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);

	cudaEventElapsedTime(&gpu_time[0], start1, stop1);

	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2);

	/*Call Kernel*/
	copy_data_to_shadowBuffer_on_Kernel <<<1, ComMaxGroupSignalCnt * Signals_Factor>>> (d_GroupSignalLocal, SignalData, SignalId);
	cudaDeviceSynchronize();

	cudaEventRecord(stop2);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&gpu_time[0], start2, stop2);

	cudaEvent_t start3, stop3;
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	cudaEventRecord(start3);

	cudaMemcpy(ComGroupSignalLocal.ComBufferRef, d_GroupSignalLocal, h_Length * sizeof(uint8), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop3);
	cudaEventSynchronize(stop3);
	cudaEventElapsedTime(&gpu_time[1], start3, stop3);




	// Return the results to the signal

}

/*********************************************************************************************************************************
 Service name:               Com_UpdateShadowSignal_GPU_ARRAY
 Service ID:                    0x0c
 Parameters (in):           SignalId--> Id of group signal to be updated.(unit16)
							SignalDataPtr --> Reference to the group signal data to be updated.
 Parameters (inout):            None
 Parameters (out):              None
 Return value:                  None
 Description:        The service Com_UpdateShadowSignal updates a group signal with the data referenced by SignalDataPtr.
 *******************************************************************************************************************************/
__global__ void copy_data_to_shadowBuffer_on_Kernel_ARRAY(uint8* d_signalgroup_in, uint8* SignalDataPtr)
{
	uint8 ThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;

	/* Check that the group signal ID is a valid ID*/

	if (ThreadIndex < ComMaxGroupSignalCnt)
	{
		/* Copy group signal to group signal buffer */
		d_signalgroup_in[ThreadIndex] = SignalDataPtr[ThreadIndex];
	}


}

void Com_UpdateShadowSignal_GPU_ARRAY(uint8* SignalIds, uint8* SignalDataPtr)
{
	Com_GroupSignalType ComGroupSignalLocal;
	/*allocate CPU memory*/
	uint8* c = (uint8*)malloc(ComMaxGroupSignalCnt * sizeof(uint8));

	/*GPU Variables*/
	uint8* d_GroupSignalLocal;
	uint8* d_SignalDataPtr;

	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1);

	/*allocate GPU memory*/
	cudaMalloc((void**)&d_GroupSignalLocal, ComMaxGroupSignalCnt * sizeof(uint8));
	cudaMalloc((void**)&d_SignalDataPtr, ComMaxGroupSignalCnt * sizeof(uint8));

	cudaMemcpy(d_SignalDataPtr, SignalDataPtr, ComMaxGroupSignalCnt * sizeof(uint8), cudaMemcpyHostToDevice);
	cudaMemcpy(d_GroupSignalLocal, c, ComMaxGroupSignalCnt * sizeof(uint8), cudaMemcpyHostToDevice);
	
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);

	cudaEventElapsedTime(&gpu_time[0], start1, stop1);

	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2);

	/*Call Kernel*/
	copy_data_to_shadowBuffer_on_Kernel_ARRAY <<< 1, ComMaxGroupSignalCnt >>> (d_GroupSignalLocal, d_SignalDataPtr);

	cudaDeviceSynchronize();

	cudaEventRecord(stop2);
	cudaEventSynchronize(stop2);

	cudaEventElapsedTime(&gpu_time[1], start2, stop2);

	cudaEvent_t start3, stop3;
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);
	cudaEventRecord(start3);

	// Return the results to the signal
	cudaMemcpy(c, d_GroupSignalLocal, ComMaxGroupSignalCnt * sizeof(uint8), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop3);
	cudaEventSynchronize(stop3);
	
	cudaEventElapsedTime(&gpu_time[2], start3, stop3);


	int len = sizeof(c)/sizeof(uint8);

	for (uint8 i = 0; i < 5; i++)
	{
		*Com.ComConfig.ComGroupSignal[SignalIds[i]].ComBufferRef = c[i];
	}

}

_global_ void Packing(uint8* ComSignalLocal_GPU, uint8* ComIPduLocal_GPU, uint8 ComBitSize_GPU, uint8 ComBitPosition_GPU) {
	uint32 Index = threadIdx.x + blockIdx.x * blockDim.x;
	if (Index >= 0 && Index < (ComBitSize_GPU)) {
		if (ComSignalLocal_GPU[Index / 8] >> (Index % 8) & 1) {
			atomicOr((int*)&ComIPduLocal_GPU[(Index + ComBitPosition_GPU) / 8], (int)1 << ((Index + ComBitPosition_GPU) % 8));
			//ComIPduLocal_GPU[(Index + ComBitPosition_GPU) / 8] |= 1 << ((Index + ComBitPosition_GPU)%8);
		}
		else {
			atomicAnd((int*)&ComIPduLocal_GPU[(Index + ComBitPosition_GPU) / 8], (int)~(1 << ((Index + ComBitPosition_GPU) % 8)));
			//ComIPduLocal_GPU[(Index + ComBitPosition_GPU) / 8] &= ~(1 << ((Index + ComBitPosition_GPU)%8));
		}
		//ComIPduLocal_GPU[ComUpdateBitPosition_GPU / 8] |= 1 << (ComUpdateBitPosition_GPU%8);
		//*ComTeamSignalUpdated_GPU = true;
	}
}

uint8 Com_SendSignal_GPU(Com_SignalIdType SignalId, const void* SignalDataPtr)
{
	uint8 ComIPduIndex, ComSignalIndex, BitIndex;
	_boolean ComCopySignal = false;
	/*
	 [SWS_Com_00804] ⌈Error code if any other API service, except Com_GetStatus, is called before the AUTOSAR COM module was initialized with Com_Init
	 or after a call to Com_Deinit:
	 error code: COM_E_UNINIT
	 value [hex]: 0x02
	 (SRS_BSW_00337)
   */
	if (ComState == COM_UNINIT)
	{
#if ComConfigurationUseDet == true
		Det_ReportError(COM_MODULE_ID, COM_INSTANCE_ID, 0x0A, COM_E_UNINIT);
#endif
		return COM_SERVICE_NOT_AVAILABLE;
	}

	/*
	  [SWS_Com_00803] ⌈API service called with wrong parameter:
	  error code: COM_E_PARAM
	  value [hex]: 0x01
	  (SRS_BSW_00337)
   */
	else if (SignalId >= ComMaxSignalCnt)
	{
#if ComConfigurationUseDet == true
		Det_ReportError(COM_MODULE_ID, COM_INSTANCE_ID, 0x0A, COM_E_PARAM);
#endif
		return COM_SERVICE_NOT_AVAILABLE;
	}

	/*
	  [SWS_Com_00805] ⌈NULL pointer checking:
	  error code: COM_E_PARAM_POINTER
	  value [hex]: 0x03
	  (SRS_BSW_00414)
	*/
	else if (SignalDataPtr == NULL)
	{
#if ComConfigurationUseDet == true
		Det_ReportError(COM_MODULE_ID, COM_INSTANCE_ID, 0x0A, COM_E_PARAM_POINTER);
#endif
		return COM_SERVICE_NOT_AVAILABLE;
	}

	else
	{
		Com_SignalType* ComSignalLocal;
		Com_IPduType* ComIPduLocal;
		ComTeamIPdu_Type* ComTeamIPduLocal;
		/*
		 * [SWS_Com_00625] ⌈If the updated signal has the ComTransferProperty TRIG-GERED and it is assigned to an I-PDU with ComTxModeMode DIRECT or MIXED,
		   then Com_SendSignal shall perform an immediate transmission (within the next main function at the latest)
		   of that I-PDU, unless the sending is delayed or prevented by other COM mechanisms.⌋ (SRS_Com_02037)
		 */


		ComSignalLocal = &Com.ComConfig.ComSignal[SignalId];
		ComIPduLocal = &Com.ComConfig.ComIPdu[ComSignalLocal->ComIpduHandler];

		ComTeamIPduLocal = &ComTeamConfig.ComTeamIPdu[ComSignalLocal->ComIpduHandler];
		ComTeamIPduLocal->ComTeamTxMode.ComTeamTxModeRepetitionPeriod = 0;

		switch (ComSignalLocal->ComTransferProperty) {
			/*[SWS_Com_00330] At any send request of a signal with ComTransferProperty TRIGGERED assigned to an I-PDU with ComTxModeMode DIRECT or MIXED, the AUTOSAR COM module shall immediately
				(within the next main function at the lat-est) initiate ComTxModeNumberOfRepetitions plus one transmissions
				of the as-signed I-PDU.
				(SRS_Com_02083)
				*/
		case PENDING:
			ComCopySignal = true;
			break;
		case TRIGGERED:
			ComCopySignal = true;
			break;
		default:
			return COM_SERVICE_NOT_AVAILABLE;

		}
		if (ComCopySignal) {
			memcpy(ComSignalLocal->ComBufferRef, SignalDataPtr, ComSignalLocal->ComSignalLength);
			uint8 IPdu_length = ComIPduLocal->ComIPduLength;
			uint8 signal_length = ComSignalLocal->ComSignalLength;
			uint8* ComSignalLocal_d;
			uint8* ComIPduLocal_d;
			uint16 output_size = ComSignalLocal->ComBitSize;

			uint8 ComBitSize_d = ComSignalLocal->ComBitSize;
			uint8 ComBitPosition_d = ComSignalLocal->ComBitPosition;
			uint8 ComUpdateBitPosition_d = ComSignalLocal->ComUpdateBitPosition;

			cudaMalloc(&ComSignalLocal_d, signal_length * sizeof(uint8));
			cudaMalloc(&ComIPduLocal_d, IPdu_length * sizeof(uint8));


			cudaMemcpy(ComSignalLocal_d, ComSignalLocal->ComBufferRef, signal_length * sizeof(uint8), cudaMemcpyHostToDevice);
			cudaMemcpy(ComIPduLocal_d, ComIPduLocal->ComBufferRef, IPdu_length * sizeof(uint8), cudaMemcpyHostToDevice);

			Packing << <ceil((ComSignalLocal->ComBitSize) / (float)8), 8 >> > (ComSignalLocal_d, ComIPduLocal_d, ComBitSize_d, ComBitPosition_d);

			cudaMemcpy(ComIPduLocal->ComBufferRef, ComIPduLocal_d, IPdu_length * sizeof(uint8), cudaMemcpyDeviceToHost);
			ComIPduLocal->ComBufferRef[ComSignalLocal->ComUpdateBitPosition / 8] |= 1 << (ComSignalLocal->ComUpdateBitPosition % 8);


		}
		else {

		}
		return E_OK;

	}

	return COM_SERVICE_NOT_AVAILABLE;
}
