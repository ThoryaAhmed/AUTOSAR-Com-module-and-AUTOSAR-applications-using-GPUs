/*******************************************************************************************************************************
FileName:                                               Com_Cfg.c
AUTOSAR Version:                                          4.2.2
******************************************************************************************************************************/
/******************************************************************************************************************************
 **                                                     Includes                                                             **
 ******************************************************************************************************************************/
#include "../inc/Com.h"
#include "Com_Cfg.h"
#include "../Inc/Com_Types.h"
#include "../Platform_Types.h"
#include "../MainWindow.h"
 /*****************************************************************************************************************************
  **                                         Post-Build Configuration variables values                                       **
  *****************************************************************************************************************************/

extern Ui::MainWindowClass* ui_extern;

/* ComSignal Buffers */
uint8 ComSignal0Buffer[1];
uint8 ComSignal1Buffer[1];
uint8 ComSignal2Buffer[1];
uint8 ComSignal3Buffer[1];
uint8 ComSignal4Buffer[1];
uint8 ComSignal5Buffer[1];
uint8 ComSignal6Buffer[1];
uint8 ComSignal7Buffer[1]; //8
uint8 ComSignal8Buffer[1]; //4

// /* ComGroupSignal Buffers */
// uint8 ComGroupSignal0Buffer[2];
// uint8 ComGroupSignal1Buffer[2];

/* Com IPdu Buffers */
uint8 ComIPdu0Buffer[2]; // signals 1 bit
uint8 ComIPdu1Buffer[2]; // signals of speed and tachometer

void Com_CbkSignal0TxAck(void)
{
}

void Com_CbkSignal1TxAck(void)
{
    //UARTprintf("Com_CbkSignal1TxAck\n");
}
void Com_CbkSignal2TxAck(void)
{
    //printf("Com_CbkSignal0TxAck\n");
}

void Com_CbkSignal3TxAck(void)
{
    //UARTprintf("Com_CbkSignal1TxAck\n");
}
void Com_CbkSignal4TxAck(void)
{
    //printf("Com_CbkSignal0TxAck\n");
}

void Com_CbkSignal5TxAck(void)
{
    //UARTprintf("Com_CbkSignal1TxAck\n");
}
void Com_CbkSignal6TxAck(void)
{
    //printf("Com_CbkSignal0TxAck\n");
}

void Com_CbkSignal7TxAck(void)
{
    //UARTprintf("Com_CbkSignal1TxAck\n");
}
void Com_CbkSignal8TxAck(void)
{
    //printf("Com_CbkSignal0TxAck\n");
}




Com_Type Com =
{
    .ComConfig =
    {
        .ComIPdu =
        {
            {
               .ComIPduDirection = Send,
               .ComIPduHandleId = 0,
               .ComIPduSignalProcessing = IMMEDIATE,
               .ComIPduType = NORMAL, // can be omiited
               .ComIPduGroupRef =
                {
                    &Com.ComConfig.ComIPduGroup[0],
                    NULL
                },
               .ComIPduSignalGroupRef =
                {
                    &Com.ComConfig.ComSignalGroup[0],
                    NULL
                },
               .ComIPduSignalRef =
                {
                    &Com.ComConfig.ComSignal[0], //fuel
                    &Com.ComConfig.ComSignal[1], //opendoor
                    &Com.ComConfig.ComSignal[2], //light
                    &Com.ComConfig.ComSignal[3], //battery
                    &Com.ComConfig.ComSignal[4], //fix
                    &Com.ComConfig.ComSignal[5], //left
                    &Com.ComConfig.ComSignal[6], //right
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
               .ComIPduDirection = Send,
               .ComIPduHandleId = 1,
               .ComIPduSignalProcessing = IMMEDIATE,
               .ComIPduType = NORMAL, // can be omiited
               .ComIPduGroupRef =
                {
                    &Com.ComConfig.ComIPduGroup[1],
                    NULL
                },
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
                .ComBufferRef = ComIPdu1Buffer,
                .ComIPduLength = 2
            }
        },

        .ComSignal =
        {
            { //
             .ComBitPosition = 0, // position within ipdu
             .ComBitSize = 1,	  // size in bits
             .ComHandleId = 0,
             .ComNotification = &Com_CbkSignal0TxAck, // only in sender side we may not need it
             .ComSignalEndianness = LITTLE_ENDIAN, // can be omitted	  // 
             .ComSignalLength = 1, // size in bytes
             .ComSignalType = UINT8,// type of signal
             .ComTransferProperty = TRIGGERED, // 
             .ComUpdateBitPosition = 7, // can be omitted
             .ComSignalInitValue = COM_SIGNAL_INIT_VALUE, //initial value
             .ComBufferRef = ComSignal0Buffer // value stored of the signal
            },
            { //1
             .ComBitPosition = 1,
             .ComBitSize = 1,
             .ComHandleId = 1,
             .ComNotification = &Com_CbkSignal1TxAck,
             .ComSignalEndianness = LITTLE_ENDIAN,
             .ComSignalLength = 1,
             .ComSignalType = UINT8,
             .ComTransferProperty = TRIGGERED,
             .ComUpdateBitPosition = 8,
             .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
             .ComBufferRef = ComSignal1Buffer
            },
            { //2
             .ComBitPosition = 2,
             .ComBitSize = 1,
             .ComHandleId = 2,
             .ComNotification = &Com_CbkSignal2TxAck,
             .ComSignalEndianness = LITTLE_ENDIAN,
             .ComSignalLength = 1,
             .ComSignalType = UINT8,
             .ComTransferProperty = TRIGGERED,
             .ComUpdateBitPosition = 9,
             .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
             .ComBufferRef = ComSignal2Buffer
            },
            { //3
             .ComBitPosition = 3,
             .ComBitSize = 3,
             .ComHandleId = 3,
             .ComNotification = &Com_CbkSignal3TxAck,
             .ComSignalEndianness = LITTLE_ENDIAN,
             .ComSignalLength = 1,
             .ComSignalType = UINT8,
             .ComTransferProperty = TRIGGERED,
             .ComUpdateBitPosition = 10,
             .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
             .ComBufferRef = ComSignal3Buffer
            },
            { //
             .ComBitPosition = 4,
             .ComBitSize = 1,
             .ComHandleId = 4,
             .ComNotification = &Com_CbkSignal4TxAck,
             .ComSignalEndianness = LITTLE_ENDIAN,
             .ComSignalLength = 11,
             .ComSignalType = UINT8,
             .ComTransferProperty = TRIGGERED,
             .ComUpdateBitPosition = 11,
             .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
             .ComBufferRef = ComSignal4Buffer
            },
            {
             .ComBitPosition = 5,
             .ComBitSize = 1,
             .ComHandleId = 5,
             .ComNotification = &Com_CbkSignal5TxAck,
             .ComSignalEndianness = LITTLE_ENDIAN,
             .ComSignalLength = 12,
             .ComSignalType = UINT8,
             .ComTransferProperty = TRIGGERED,
             .ComUpdateBitPosition = 12,
             .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
             .ComBufferRef = ComSignal5Buffer
            },
            {
             .ComBitPosition = 6,
             .ComBitSize = 1,
             .ComHandleId = 6,
             .ComNotification = &Com_CbkSignal6TxAck,
             .ComSignalEndianness = LITTLE_ENDIAN,
             .ComSignalLength = 13,
             .ComSignalType = UINT8,
             .ComTransferProperty = TRIGGERED,
             .ComUpdateBitPosition = 13,
             .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
             .ComBufferRef = ComSignal6Buffer
            },
            {
             .ComBitPosition = 0,
             .ComBitSize = 8,
             .ComHandleId = 7,
             .ComNotification = &Com_CbkSignal7TxAck,
             .ComSignalEndianness = LITTLE_ENDIAN,
             .ComSignalLength = 1,
             .ComSignalType = UINT8,
             .ComTransferProperty = PENDING,
             .ComUpdateBitPosition = 14,
             .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
             .ComBufferRef = ComSignal7Buffer
            },
            { //Tachomemter

             .ComBitPosition = 8,
             .ComBitSize = 4,
             .ComHandleId = 8,
             .ComNotification = &Com_CbkSignal8TxAck,
             .ComSignalEndianness = LITTLE_ENDIAN,
             .ComSignalLength = 1,
             .ComSignalType = UINT8,
             .ComTransferProperty = PENDING,
             .ComUpdateBitPosition = 15,
             .ComSignalInitValue = COM_SIGNAL_INIT_VALUE,
             .ComBufferRef = ComSignal8Buffer
            }
        }
    }


};
