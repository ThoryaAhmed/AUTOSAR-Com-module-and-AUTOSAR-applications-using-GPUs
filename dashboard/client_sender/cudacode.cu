#include "cudacode.cuh"
#include "inc/ComTeam_Types.h"
#include "Std_Types.h"
#include "stdio.h"
#include "Gen/Com_Cfg.h"
#include "inc/ComStack_Cfg.h"

extern Com_Type Com;
extern ComState_Type ComState;

__global__ void Packing(uint8* ComSignalLocal_GPU, uint8* ComIPduLocal_GPU, uint8 ComBitSize_GPU, uint8 ComBitPosition_GPU) {
	uint32 Index = threadIdx.x + blockIdx.x * blockDim.x;
	if (Index >= 0 && Index < (ComBitSize_GPU)) {
		if (ComSignalLocal_GPU[Index / 8] >> (Index % 8) & 1) {
			atomicOr((int*)&ComIPduLocal_GPU[(Index + ComBitPosition_GPU) / 32], (int)1 << ((Index + ComBitPosition_GPU) % 32));
			//ComIPduLocal_GPU[(Index + ComBitPosition_GPU) / 8] |= 1 << ((Index + ComBitPosition_GPU)%8);
		}
		else {
			atomicAnd((int*)&ComIPduLocal_GPU[(Index + ComBitPosition_GPU) / 32], (int)~(1 << ((Index + ComBitPosition_GPU) % 32)));
			//ComIPduLocal_GPU[(Index + ComBitPosition_GPU) / 8] &= ~(1 << ((Index + ComBitPosition_GPU)%8));
		}
		//ComIPduLocal_GPU[ComUpdateBitPosition_GPU / 8] |= 1 << (ComUpdateBitPosition_GPU%8);
		//*ComTeamSignalUpdated_GPU = true;
	}
}


uint8 Com_SendSignal_GPU(Com_SignalIdType SignalId, const void* SignalDataPtr)
{
	uint8 ComIPduIndex, ComSignalIndex, BitIndex;
	boolean ComCopySignal = false;
	/*
	 [SWS_Com_00804] ?Error code if any other API service, except Com_GetStatus, is called before the AUTOSAR COM module was initialized with Com_Init
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
	  [SWS_Com_00803] ?API service called with wrong parameter:
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
	  [SWS_Com_00805] ?NULL pointer checking:
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
		 * [SWS_Com_00625] ?If the updated signal has the ComTransferProperty TRIG-GERED and it is assigned to an I-PDU with ComTxModeMode DIRECT or MIXED,
		   then Com_SendSignal shall perform an immediate transmission (within the next main function at the latest)
		   of that I-PDU, unless the sending is delayed or prevented by other COM mechanisms.? (SRS_Com_02037)
		 */


		ComSignalLocal = &Com.ComConfig.ComSignal[SignalId];

		if (SignalId < 7)
			ComIPduLocal = &Com.ComConfig.ComIPdu[0];
		else
			ComIPduLocal = &Com.ComConfig.ComIPdu[1];



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
			uint8* h_data = 0;

			memcpy(ComSignalLocal->ComBufferRef, SignalDataPtr, ComSignalLocal->ComSignalLength);
			uint8 IPdu_length = ComIPduLocal->ComIPduLength;
			uint8 signal_length = ComSignalLocal->ComSignalLength;
			uint8* ComSignalLocal_d;
			uint8* ComIPduLocal_d;
			uint8* d_data;

			uint8 ComBitSize_d = ComSignalLocal->ComBitSize;
			uint8 ComBitPosition_d = ComSignalLocal->ComBitPosition;
			uint8 ComUpdateBitPosition_d = ComSignalLocal->ComUpdateBitPosition;

			cudaEvent_t start1, stop1;
			cudaEventCreate(&start1);
			cudaEventCreate(&stop1);
			cudaEventRecord(start1);

			cudaMalloc(&ComSignalLocal_d, signal_length * sizeof(uint8));
			cudaMalloc(&ComIPduLocal_d, IPdu_length * sizeof(uint8));

			cudaMemcpyAsync(ComSignalLocal_d, ComSignalLocal->ComBufferRef, signal_length * sizeof(uint8), cudaMemcpyHostToDevice);
			cudaMemcpyAsync(ComIPduLocal_d, ComIPduLocal->ComBufferRef, IPdu_length * sizeof(uint8), cudaMemcpyHostToDevice);


			Packing << <ceil((ComSignalLocal->ComBitSize) / (float)8), 8 >> > (ComSignalLocal_d, ComIPduLocal_d, ComBitSize_d, ComBitPosition_d);

			cudaDeviceSynchronize();

			cudaMemcpyAsync(ComIPduLocal->ComBufferRef, ComIPduLocal_d, IPdu_length * sizeof(uint8), cudaMemcpyDeviceToHost);

			ComIPduLocal->ComBufferRef[ComSignalLocal->ComUpdateBitPosition / 8] |= 1 << (ComSignalLocal->ComUpdateBitPosition % 8);


		}
		else {

		}
		return E_OK;

	}

	return COM_SERVICE_NOT_AVAILABLE;
}

