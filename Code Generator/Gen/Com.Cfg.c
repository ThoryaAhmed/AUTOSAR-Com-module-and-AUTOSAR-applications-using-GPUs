/*******************************************************************************************************************************
FileName:                                               Com_Cfg.c
AUTOSAR Version:                                          4.2.2
******************************************************************************************************************************/
/******************************************************************************************************************************
 **                                                     Includes                                                             **
 ******************************************************************************************************************************/
#include "Inc/Com.h"
#include "Com_Cfg.h"
#include "Inc/Com_Types.h"
#include "Platform_Types.h"
 /*****************************************************************************************************************************
  **                                         Post-Build Configuration variables values                                       **
  *****************************************************************************************************************************/
/* ComSignal Buffers */
uint8 ComSignal0Buffer[1] = { 0 };
uint8 ComSignal1Buffer[1];
uint8 ComSignal2Buffer[1];
uint8 ComSignal3Buffer[1];
uint8 ComSignal4Buffer[1];
uint8 ComSignal5Buffer[1];
uint8 ComSignal6Buffer[1];
uint8 ComSignal7Buffer[1]; //8
uint8 ComSignal8Buffer[1]; //4

 /* ComGroupSignal Buffers */
 uint8 ComGroupSignal0Buffer[1];
 uint8 ComGroupSignal1Buffer[1];
 uint8 ComGroupSignal2Buffer[1];
 uint8 ComGroupSignal3Buffer[1];

/* Com IPdu Buffers */
uint8 ComIPdu0Buffer[8]; // signals 1 bit
uint8 ComIPdu1Buffer[8];// signals of speed and tachometer

void Com_CbkSignal0RxAck(void)
{
}

void Com_CbkSignal1RxAck(void)
{
    //UARTprintf("Com_CbkSignal1TxAck\n");
}
void Com_CbkSignal2RxAck(void)
{
    //printf("Com_CbkSignal0TxAck\n");
}

void Com_CbkSignal3RxAck(void)
{
    //UARTprintf("Com_CbkSignal1TxAck\n");
}
void Com_CbkSignal4RxAck(void)
{
    //printf("Com_CbkSignal0TxAck\n");
}

void Com_CbkSignal5RxAck(void)
{
    //UARTprintf("Com_CbkSignal1TxAck\n");
}
void Com_CbkSignal6RxAck(void)
{
    //printf("Com_CbkSignal0TxAck\n");
}

void Com_CbkSignal7RxAck(void)
{
    //UARTprintf("Com_CbkSignal1TxAck\n");
}
void Com_CbkSignal8RxAck(void)
{
    //printf("Com_CbkSignal0TxAck\n");
}




Com_Type Com =
{
    .ComConfig=
    {
        .ComIPdu=
        { 
            {
               .ComIPduDirection = send,
               .ComIPduHandleId= 0,
               .ComIPduSignalProcessing = IMMEDIATE,
               .ComIPduType = NORMAL,
               .ComIPduSignalGroupRef = 
               { 
                    &Com.ComConfig.ComSignalGroup[0],
                    NULL 
               },
               .ComIPduSignalRef = 
               { 
                    &Com.ComConfig.ComSignal[0],
                     
                    &Com.ComConfig.ComSignal[1],
                     
                    &Com.ComConfig.ComSignal[2],
                     
                    &Com.ComConfig.ComSignal[3],
                     
                    &Com.ComConfig.ComSignal[4],
                     
                    &Com.ComConfig.ComSignal[5],
                     
                    &Com.ComConfig.ComSignal[6],
                    NULL 
               },
               
               .ComTxIPdu =
               {
                    .ComMinimumDelayTime = 0.5,
                    .ComTxIPduClearUpdateBit = Confirmation,
                    .ComTxIPduUnusedAreasDefault = 255,
                    .ComTxModeFalse =
                    {
                        .ComTxMode =
                        {
                            .ComTxModeMode = PERIODIC,
                            .ComTxModeNumberOfRepetitions = 2,
                            .ComTxModeRepetitionPeriod = 2,
                            .ComTxModeTimePeriod = 2
                        }
                    }
               },
               
               .ComBufferRef = ComIPdu0Buffer,
               .ComIPduLength = 2
            }, 
            {
               .ComIPduDirection = Receive,
               .ComIPduHandleId= 1,
               .ComIPduSignalProcessing = DEFERRED,
               .ComIPduType = NORMAL,
               .ComIPduSignalGroupRef = 
               { 
                    &Com.ComConfig.ComSignalGroup[1],
                    NULL 
               },
               .ComIPduSignalRef = 
               { 
                    &Com.ComConfig.ComSignal[7],
                     
                    &Com.ComConfig.ComSignal[8],
                    NULL 
               },
               
               .ComBufferRef = ComIPdu1Buffer,
               .ComIPduLength = 5
            } 
        }
        .ComSignal=
        { 
            {
                .ComBitPosition = 0,
                .ComBitSize= 1,
                .ComHandleId = 0,
                .ComSignalLength = 1,
                .ComSignalType= __UINT8,
                .ComTransferProperty =  TRIGGERED,
                .ComUpdateBitPosition = 7,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComSignal0Buffer,
                .ComIpduHandler = 0
            }, 
            {
                .ComBitPosition = 1,
                .ComBitSize= 1,
                .ComHandleId = 1,
                .ComSignalLength = 1,
                .ComSignalType= __UINT8,
                .ComTransferProperty =  TRIGGERED,
                .ComUpdateBitPosition = 8,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComSignal1Buffer,
                .ComIpduHandler = 0
            }, 
            {
                .ComBitPosition = 2,
                .ComBitSize= 1,
                .ComHandleId = 2,
                .ComSignalLength = 1,
                .ComSignalType= __UINT8,
                .ComTransferProperty =  TRIGGERED,
                .ComUpdateBitPosition = 9,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComSignal2Buffer,
                .ComIpduHandler = 0
            }, 
            {
                .ComBitPosition = 3,
                .ComBitSize= 1,
                .ComHandleId = 3,
                .ComSignalLength = 1,
                .ComSignalType= __UINT8,
                .ComTransferProperty =  TRIGGERED,
                .ComUpdateBitPosition = 10,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComSignal3Buffer,
                .ComIpduHandler = 0
            }, 
            {
                .ComBitPosition = 4,
                .ComBitSize= 1,
                .ComHandleId = 4,
                .ComSignalLength = 1,
                .ComSignalType= __UINT8,
                .ComTransferProperty =  TRIGGERED,
                .ComUpdateBitPosition = 11,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComSignal4Buffer,
                .ComIpduHandler = 0
            }, 
            {
                .ComBitPosition = 5,
                .ComBitSize= 1,
                .ComHandleId = 5,
                .ComSignalLength = 1,
                .ComSignalType= __UINT8,
                .ComTransferProperty =  TRIGGERED,
                .ComUpdateBitPosition = 12,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComSignal5Buffer,
                .ComIpduHandler = 0
            }, 
            {
                .ComBitPosition = 6,
                .ComBitSize= 1,
                .ComHandleId = 6,
                .ComSignalLength = 1,
                .ComSignalType= __UINT8,
                .ComTransferProperty =  TRIGGERED,
                .ComUpdateBitPosition = 13,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComSignal6Buffer,
                .ComIpduHandler = 0
            }, 
            {
                .ComBitPosition = 0,
                .ComBitSize= 8,
                .ComHandleId = 7,
                .ComSignalLength = 1,
                .ComSignalType= __UINT8,
                .ComTransferProperty =  PENDING,
                .ComUpdateBitPosition = 12,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComSignal7Buffer,
                .ComIpduHandler = 1
            }, 
            {
                .ComBitPosition = 8,
                .ComBitSize= 4,
                .ComHandleId = 8,
                .ComSignalLength = 1,
                .ComSignalType= __UINT8,
                .ComTransferProperty =  PENDING,
                .ComUpdateBitPosition = 13,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComSignal8Buffer,
                .ComIpduHandler = 1
            } 
        },
        .ComGroupSignal=
        { 
            {
                .ComBitPosition = 0,
                .ComBitSize = 8,
                .ComHandleId = 0,
                .ComSignalLength = 1,
                .ComSignalType = __UINT8,
                .ComTransferProperty = PENDING,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComGroupSignal0Buffer
            }, 
            {
                .ComBitPosition = 8,
                .ComBitSize = 4,
                .ComHandleId = 1,
                .ComSignalLength = 1,
                .ComSignalType = __UINT8,
                .ComTransferProperty = PENDING,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComGroupSignal1Buffer
            }, 
            {
                .ComBitPosition = 8,
                .ComBitSize = 8,
                .ComHandleId = 2,
                .ComSignalLength = 1,
                .ComSignalType = __UINT8,
                .ComTransferProperty = TRIGGERED,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComGroupSignal2Buffer
            }, 
            {
                .ComBitPosition = 15,
                .ComBitSize = 8,
                .ComHandleId = 3,
                .ComSignalLength = 1,
                .ComSignalType = __UINT8,
                .ComTransferProperty = TRIGGERED,
                .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
                .ComBufferRef = ComGroupSignal3Buffer
            } 
        },
        .ComSignalGroup =
        { 
            {
                .ComHandleId = 0,
                .ComNotification = NULL,
                .ComTransferProperty = TRIGGERD,
                .ComUpdateBitPosition = 34,
                .ComGroupSignalRef = 
                { 
                    &Com.ComConfig.ComGroupSignal[groupsignalref],
                     
                    &Com.ComConfig.ComGroupSignal[groupsignalref],
                    NULL 
                },
                .ComGroupSignalsNumbers = 2
            }, 
            {
                .ComHandleId = 1,
                .ComNotification = NULL,
                .ComTransferProperty = TRIGGERD,
                .ComUpdateBitPosition = 34,
                .ComGroupSignalRef = 
                { 
                    &Com.ComConfig.ComGroupSignal[groupsignalref],
                    NULL 
                },
                .ComGroupSignalsNumbers = 1
            } 
        },
    }
};
         