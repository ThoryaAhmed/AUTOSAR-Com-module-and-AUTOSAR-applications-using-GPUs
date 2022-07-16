/*******************************************************************************************************************************
FileName:                                               Com_Cfg.c
AUTOSAR Version:                                          4.2.2
******************************************************************************************************************************/
/******************************************************************************************************************************
 **                                                     Includes                                                             **
 ******************************************************************************************************************************/
#pragma once

#include <math.h>

#include "Com_Cfg.cuh"
#include "../inc/Com_Types.cuh"

// maybe they should be in the .cuh
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*****************************************************************************************************************************
 **                                         Post-Build Configuration variables values                                       **
 *****************************************************************************************************************************/

/* ComSignal Buffers */
__device__ uint8 image_signal_buffer[WIDTH * HEIGHT * CHANNEL_NUM];

/* Com IPdu Buffers */
__device__ uint8 image_IPDU_buffer[WIDTH * HEIGHT * CHANNEL_NUM + 1];

__device__ Com_Type Com_GPU[1] =
    {
        {//.ComConfig =
         {
             // .ComSignal =
             {
                 {
                     //
                     12,                                              // .ComBitPosition =  // try 25
                     WIDTH *HEIGHT *CHANNEL_NUM * 8 * FACTOR *FACTOR, // .ComBitSize =
                     0,                                               // .ComHandleId =
                     NULL,                                            // .ComNotification =
                     LITTLE_ENDIAN,                                   // .ComSignalEndianness =
                     WIDTH *FACTOR *HEIGHT *FACTOR *CHANNEL_NUM,      // .ComSignalLength =
                     _UINT8,                                          // .ComSignalType =
                     TRIGGERED,                                       // .ComTransferProperty =
                     WIDTH *HEIGHT *CHANNEL_NUM * 8 * FACTOR *FACTOR, // .ComUpdateBitPosition =
                     COM_SIGNAL_INIT_VALUE,                           // .ComSignalInitValue =
                     image_signal_buffer,                             // .ComBufferRef =
                     0                                                // .ComIpduHandler =
                 }},

             //.ComIPdu =
             {
                 {
                     Receive,   // .ComIPduDirection =
                     0,         // .ComIPduHandleId =
                     IMMEDIATE, // .ComIPduSignalProcessing =
                     TP,        // .ComIPduType =
                                // .ComIPduGroupRef =
                     {
                         NULL},
                     // .ComIPduSignalGroupRef =
                     {
                         NULL},
                     // .ComIPduSignalRef =
                     {
                         // &Com.ComConfig.ComSignal[0], // causes circular dependency
                         NULL},
                     // .ComTxIPdu =
                     {
                         0.5,          //.ComMinimumDelayTime =
                         Confirmation, // .ComTxIPduClearUpdateBit =
                         255,          // .ComTxIPduUnusedAreasDefault =
                                       // .ComTxModeFalse =
                         {
                             // .ComTxMode =
                             {
                                 PERIODIC, // .ComTxModeMode =
                                 2,        // .ComTxModeNumberOfRepetitions =
                                 2,        // .ComTxModeRepetitionPeriod =
                                 2         // .ComTxModeTimePeriod =
                             }}},
                     image_IPDU_buffer,             // .ComBufferRef =
                     WIDTH *HEIGHT *CHANNEL_NUM + 1 // .ComIPduLength =
                 }}}}};
