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

