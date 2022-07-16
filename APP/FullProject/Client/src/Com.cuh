#pragma once

#include "./inc/Com_Types.cuh"
#include <qstring.h>


#define WIDTH 1280
#define HEIGHT 720
#define CHANNEL_NUM 3
#define FACTOR 0.5

void Com_Init();
QString QCom_ReceiveSignal_GPU(Com_SignalIdType SignalId, void* SignalDataPtr);
uint8 Com_ReceiveSignal_GPU(Com_SignalIdType SignalId, void* SignalDataPtr);