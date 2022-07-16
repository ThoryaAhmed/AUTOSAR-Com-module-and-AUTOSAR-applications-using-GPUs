/*******************************************************************************************************************************
FileName:                                               Com_Cfg.cuh
AUTOSAR Version:                                          4.2.2
******************************************************************************************************************************/
#pragma once

/*******************************************************************************************************************************
**                                                                        Defines                                                                                **
********************************************************************************************************************************/
#define WIDTH 1280
#define HEIGHT 720
#define CHANNEL_NUM 3
#define FACTOR 1 // not 0.5 as the server process the full image before sending

#define COM_TX_IPDU_UNUSED_AREAS_DEFAULT    (uint8)0x00
#define COM_SIGNAL_INIT_VALUE    (uint8)0xFF
#define ComMaxSignalGroupCnt    (uint8)1 // was 20 , 1 lets only null thus zero
#define ComMaxGroupSignalCnt    (uint8)1 // was 20
#define ComMaxSignalCnt    (uint8) 1 // was 20 should be at least 2
#define ComMaxIPduGroupCnt    (uint8)4 // all IPDUs in COM



/**************************************************************************************************
**
Name:                                   ComConfigurationUseDet
Type:                                   EcucBooleanParamDef
Description:            The error hook shall contain code to call the Det.
                        If this parameter is configured COM_DEV_ERROR_DETECT shall be set
                        to ON as output of the configuration tool. (as input for the source code)                                           **
**************************************************************************************************/
#define ComConfigurationUseDet false


/**************************************************************************************************
**
Name:                                   ComEnableMDTForCyclicTransmission
Type:                                   EcucBooleanParamDef
Description:            Enables globally for the whole Com module the minimum delay time monitoring
                        for cyclic and repeated transmissions (ComTxModeMode=PERIODIC or ComTxModeMode=MIXED
                        for the cyclic transmissions,
                        ComTxModeNumberOfRepetitions > 0 for repeated transmissions).                                         **
**************************************************************************************************/
#define ComEnableMDTForCyclicTransmission false


/**************************************************************************************************
**
Name:                                   ComRetryFailedTransmitRequests
Type:                                   EcucBooleanParamDef
Description:                If this Parameter is set to true, retry of failed transmission requests
                            is enabled. If this Parameter is not present, the default value is assumed                              **
**************************************************************************************************/
#define ComRetryFailedTransmitRequests  false



/**************************************************************************************************

Name:                                ComSupportedIPduGroups
Type:                                EcucIntegerParamDef
Description:                Defines the maximum number of supported I-PDU groups.
Range:                               0 ---> 65535
**************************************************************************************************/
#define ComSupportedIPduGroups  (uint16)    2


/**************************************************************************************************
**
Name:                                   ComRxTimeBase
Type:                                   EcucFloatParamDef
Description:                The period between successive calls to Com_MainFunctionRx in seconds.
                            This parameter may be used by the COM generator to transform the values
                            of the reception related timing configuration parameters of the COM
                            module to internal implementation specific counter or tick values. The
                            COM module's internal timing handling is implementation specific.
                            The COM module (generator) may rely on the fact that
                            Com_MainFunctionRx is scheduled according to the value configuredhere
Range:                                  0 ---> 3600                                         **
**************************************************************************************************/
#define ComRxTimeBase   (float32)    0.1


/**************************************************************************************************
**
Name:                                   ComTxTimeBase

Type:                                   EcucFloatParamDef

Description:                The period between successive calls to Com_MainFunctionTx in seconds.
                            This parameter may be used by the COM generator to transform
                            the values of the transmission related timing configuration parameters of the COM
                            module to internal implementation specific counter or tick values. The
                            COM module's internal timing handling is implementation specific.
                            The COM module (generator) may rely on the fact that
                            Com_MainFunctionTx is scheduled according to the value configured here
Range:                                  0 ---> 3600
**************************************************************************************************/
#define ComTxTimeBase   (float32)    0.5


/**************************************************************************************************

Name:                                ComMaxIPduCnt
Type:                                EcucIntegerParamDef
Description:                Defines the maximum number of supported I-PDU groups.
Range:                               0 ---> 18446744073709551615
**************************************************************************************************/
#define ComMaxIPduCnt  (uint64)    1 // was 3


