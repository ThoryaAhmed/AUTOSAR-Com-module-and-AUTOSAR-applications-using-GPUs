/*********************************************************************************************************************************
 Service name:               Com_UpdateShadowSignal
 Service ID:                    0x0c
 Parameters (in):           SignalId--> Id of group signal to be updated.
                            SignalDataPtr --> Reference to the group signal data to be updated.
 Parameters (inout):            None
 Parameters (out):              None
 Return value:                  None
 Description:        The service Com_UpdateShadowSignal updates a group signal with the data referenced by SignalDataPtr.
 *******************************************************************************************************************************/
void Com_UpdateShadowSignal(Com_SignalIdType SignalId,const void* SignalDataPtr)
{
    uint8 ComGroupSignalIndex;

    /* Check that the group signal ID is a valid ID*/
    if(SignalId < ComMaxGroupSignalCnt)
    {
        /*Find GroupSignal with such ID*/
        for(ComGroupSignalIndex = 0; ComGroupSignalIndex < ComMaxGroupSignalCnt; ComGroupSignalIndex++)
        {
            if(Com.ComConfig.ComGroupSignal[ComGroupSignalIndex].ComHandleId == SignalId)
            {
                /* Copy group signal to group signal buffer */
                memcpy(Com.ComConfig.ComGroupSignal[SignalId].ComBufferRef, SignalDataPtr, Com.ComConfig.ComGroupSignal[SignalId].ComSignalLength);
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
void Com_ReceiveShadowSignal(Com_SignalIdType SignalId, void* SignalDataPtr)
{
    uint8 ComGroupSignalIndex;

    /* Check that the group signal ID is a valid ID*/
    if(SignalId < ComMaxGroupSignalCnt)
    {
        /*Find GroupSignal with such ID*/
        for(ComGroupSignalIndex = 0; ComGroupSignalIndex < ComMaxGroupSignalCnt; ComGroupSignalIndex++)
        {
            if(Com.ComConfig.ComGroupSignal[ComGroupSignalIndex].ComHandleId == SignalId)
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

uint8 Com_ReceiveSignalGroup(Com_SignalGroupIdType SignalGroupId)
{
	uint8 ComIPduIndex, ComSignalGroupIndex, ComGroupSignalIndex, BitIndex;
	Com_GroupSignalType* ComGroupSignalLocal;
	Com_IPduType* ComIPduLocal;

	/*[SWS_Com_00638] âŒˆThe service Com_ReceiveSignalGroup shall copy the received
	signal group from the I-PDU to the shadow buffer.âŒ‹ (SRS_Com_02041)*/
	if (SignalGroupId < ComMaxSignalGroupCnt)
	{
		/*After this call, the group signals could be copied from the shadow buffer to the RTE
		by calling Com_ReceiveSignal.*/
		/* Find the IPdu which contains this signal */

		ComGroupSignalLocal = &Com.ComConfig.ComGroupSignal[SignalGroupId];


		for (ComIPduIndex = 0; ComIPduIndex < ComMaxIPduCnt; ComIPduIndex++)
		{
			for (ComSignalGroupIndex = 0; Com.ComConfig.ComIPdu[ComIPduIndex].ComIPduSignalGroupRef[ComSignalGroupIndex] != NULL; ComSignalGroupIndex++)
			{
				if (Com.ComConfig.ComIPdu[ComIPduIndex].ComIPduSignalGroupRef[ComSignalGroupIndex]->ComHandleId == SignalGroupId)
				{
					/* Get IPdu */
					ComIPduLocal = &Com.ComConfig.ComIPdu[ComIPduIndex];

					for (ComGroupSignalIndex = 0; ComIPduLocal->ComIPduSignalGroupRef[ComSignalGroupIndex]->ComGroupSignalRef[ComGroupSignalIndex] != NULL; ComGroupSignalIndex++)
					{
						/*Get Group Signal*/
						ComGroupSignalLocal = ComIPduLocal->ComIPduSignalGroupRef[ComSignalGroupIndex]->ComGroupSignalRef[ComGroupSignalIndex];

						/* Write data from IPdu to GroupSignal buffer to IPdu*/
						for (BitIndex = ComGroupSignalLocal->ComBitPosition; BitIndex < ComGroupSignalLocal->ComBitPosition + ComGroupSignalLocal->ComBitSize; BitIndex++)
						{
							if ((ComIPduLocal->ComBufferRef[BitIndex / 8] >> (BitIndex % 8)) & 1)
							{
								ComGroupSignalLocal->ComBufferRef[(BitIndex - ComGroupSignalLocal->ComBitPosition) / 8] |= 1 << ((BitIndex - ComGroupSignalLocal->ComBitPosition) % 8);
							}
							else
							{
								ComGroupSignalLocal->ComBufferRef[(BitIndex - ComGroupSignalLocal->ComBitPosition) / 8] &= ~(1 << ((BitIndex - ComGroupSignalLocal->ComBitPosition) % 8));
							}
						}
						return E_OK;
					}
				}
				else
				{
				}
			}
		}
	}
	else
	{

	}
	return COM_SERVICE_NOT_AVAILABLE;
}


/*********************************************************************************************************************************
 Service name:               Com_SendSignal
 Service ID:                    0x0a
 Parameters (in):           SignalId--> Id of signal to be sent.
                            SignalDataPtr --> Reference to the signal data to be transmitted.
 Parameters (inout):            None
 Parameters (out):              None
 Return value:              uint8 --->
                                 E_OK: service has been accepted
                                 COM_SERVICE_NOT_AVAILABLE: corresponding I-PDU group
                                     was stopped (or service failed due to development error)
                                 COM_BUSY: in case the TP-Buffer is locked for large data types
                                     handling
 Description:        The service Com_SendSignal updates the signal object identified by SignalId with the signal
                     referenced by the SignalDataPtr parameter
*******************************************************************************************************************************/
uint8 Com_SendSignal(Com_SignalIdType SignalId, const void* SignalDataPtr)
{
	uint8 ComIPduIndex, ComSignalIndex, BitIndex;
	_boolean ComCopySignal = false;
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

		for (ComIPduIndex = 0; ComIPduIndex < ComMaxIPduCnt; ComIPduIndex++)
		{
			for (ComSignalIndex = 0; Com.ComConfig.ComIPdu[ComIPduIndex].ComIPduSignalRef[ComSignalIndex] != NULL; ComSignalIndex++)
			{
				if (Com.ComConfig.ComIPdu[ComIPduIndex].ComIPduSignalRef[ComSignalIndex]->ComHandleId == SignalId)
				{
					ComIPduLocal = &Com.ComConfig.ComIPdu[ComIPduIndex];
					ComSignalLocal = Com.ComConfig.ComIPdu[ComIPduIndex].ComIPduSignalRef[ComSignalIndex];
					ComTeamIPduLocal = &ComTeamConfig.ComTeamIPdu[ComIPduIndex];

					ComTeamIPduLocal->ComTeamTxMode.ComTeamTxModeRepetitionPeriod = 0;

					switch (ComSignalLocal->ComTransferProperty)
					{
						/*[SWS_Com_00330] At any send request of a signal with ComTransferProperty TRIGGERED assigned to an I-PDU with ComTxModeMode DIRECT or MIXED, the AUTOSAR COM module shall immediately
						   (within the next main function at the lat-est) initiate ComTxModeNumberOfRepetitions plus one transmissions
						   of the as-signed I-PDU.
						   (SRS_Com_02083)
						 */
					case PENDING:
						ComCopySignal = true;
						break;

					case TRIGGERED:
						ComTeamIPduLocal->ComTeamTxMode.ComTeamTxIPduNumberOfRepetitions = ComIPduLocal->ComTxIPdu.ComTxModeFalse.ComTxMode.ComTxModeNumberOfRepetitions + 1;
						ComCopySignal = true;
						break;

					default:
						return COM_SERVICE_NOT_AVAILABLE;
					}

					if (ComCopySignal)
					{
						/*Copy signal to signal buffer*/
						memcpy(ComSignalLocal->ComBufferRef, SignalDataPtr, ComSignalLocal->ComSignalLength);

						/* Write data from signal buffer to IPdu*/
						for (BitIndex = 0; BitIndex < ComSignalLocal->ComBitSize; BitIndex++)
						{
							if ((ComSignalLocal->ComBufferRef[BitIndex / 8] >> (BitIndex % 8)) & 1)
							{
								ComIPduLocal->ComBufferRef[(BitIndex + ComSignalLocal->ComBitPosition) / 8] |= 1 << ((BitIndex + ComSignalLocal->ComBitPosition) % 8);
							}
							else
							{
								ComIPduLocal->ComBufferRef[(BitIndex + ComSignalLocal->ComBitPosition) / 8] &= ~(1 << ((BitIndex + ComSignalLocal->ComBitPosition) % 8));
							}
						}
						/*Set update bit*/
						ComIPduLocal->ComBufferRef[ComSignalLocal->ComUpdateBitPosition / 8] |= 1 << (ComSignalLocal->ComUpdateBitPosition % 8);

						ComTeamConfig.ComTeamSignal[ComSignalLocal->ComHandleId].ComTeamSignalUpdated = true;
					}
					else
					{

					}

					return E_OK;
				}
				else
				{

				}
			}
		}
	}

	return COM_SERVICE_NOT_AVAILABLE;
}
