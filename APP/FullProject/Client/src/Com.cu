#include "Com.cuh"
#include "./gen/Com_Cfg.cu"

Com_Type Com_CPU[1];
static ComState_Type ComState = COM_UNINIT;

extern uint8* rgb_in;
extern uint8* h_image_out;

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

void Com_Init() {
    cudaMemcpyFromSymbol(Com_CPU, Com_GPU, sizeof(Com_Type));
    ComState = COM_READY;
}

// NOT USED IN CLIENT
uint8 Com_SendSignal_GPU(Com_SignalIdType SignalId, const void* SignalDataPtr)
{
    const void* signal_buf = Com_CPU[0].ComConfig.ComSignal->ComBufferRef;
    const void* IPDU_buf = Com_CPU[0].ComConfig.ComIPdu->ComBufferRef;
    size_t sizeInBytes = Com_CPU[0].ComConfig.ComSignal->ComSignalLength;

    uint8* ComSignalLocal = (uint8*)signal_buf;
    uint8* ComIPduLocal = (uint8*)IPDU_buf;
    uint8 ComBitSize = Com_CPU[0].ComConfig.ComSignal->ComBitSize;
    uint8 ComBitPosition = Com_CPU[0].ComConfig.ComSignal->ComBitPosition;

    cudaMemcpy(ComSignalLocal, SignalDataPtr, sizeInBytes, cudaMemcpyDeviceToDevice);

    dim3 dimGrid(ceil(WIDTH / 32.0), ceil(HEIGHT / 32.0), 1);
    dim3 dimBlock(32, 32, 1);
    copyToIpdu << <dimGrid, dimBlock >> > ();

    cudaMemcpy(h_image_out, IPDU_buf, sizeInBytes, cudaMemcpyDeviceToHost);
    return 0;
}

uint8 Com_ReceiveSignal_GPU(Com_SignalIdType SignalId, void* SignalDataPtr)
{
    if (ComState == COM_UNINIT)
        return COM_SERVICE_NOT_AVAILABLE;

    else if (SignalId >= ComMaxSignalCnt)
        return COM_SERVICE_NOT_AVAILABLE;

    else if (SignalDataPtr == NULL)
        return COM_SERVICE_NOT_AVAILABLE;

    else{
        const void* signal_buf = Com_CPU[0].ComConfig.ComSignal[SignalId].ComBufferRef;
        const int IPDU_handler = Com_CPU[0].ComConfig.ComSignal[SignalId].ComIpduHandler;
        const void* IPDU_buf = Com_CPU[0].ComConfig.ComIPdu[IPDU_handler].ComBufferRef;
        size_t sizeInBytes = Com_CPU[0].ComConfig.ComSignal->ComSignalLength;

        uint8* ComSignalLocal = (uint8*)signal_buf;
        uint8* ComIPduLocal = (uint8*)IPDU_buf;
        uint8 ComBitSize = Com_CPU[0].ComConfig.ComSignal->ComBitSize;
        uint8 ComBitPosition = Com_CPU[0].ComConfig.ComSignal->ComBitPosition;
        uint8 ComIPduLengthBits = Com_CPU[0].ComConfig.ComIPdu->ComIPduLength * 8;

        cudaMemcpy((uint8*)IPDU_buf, rgb_in, sizeInBytes, cudaMemcpyHostToDevice);

        dim3 dimGrid(ceil(WIDTH / 32.0), ceil(HEIGHT / 32.0), 1);
        dim3 dimBlock(32, 32, 1);
        copyToSig << <dimGrid, dimBlock >> > ();

        cudaMemcpy(SignalDataPtr, signal_buf, sizeInBytes, cudaMemcpyDeviceToDevice);

        return E_OK;
    }
    
    return COM_SERVICE_NOT_AVAILABLE;
}

QString QCom_ReceiveSignal_GPU(Com_SignalIdType SignalId, void* SignalDataPtr) {
    Com_ReceiveSignal_GPU(SignalId, SignalDataPtr);
    return "Done";
}