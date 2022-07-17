
/*********************************************************************************************************************************
 Service name:               Com_ReceiveSignalGroup
 Service ID:                    0x0e
 Parameters (in):               SignalGroupId -->  Id of signal group to be sent.
 Parameters (inout):            None
 Parameters (out):              None
 Return value:                  uint8 --->
                                 ->E_OK: service has been accepted
                                 ->COM_SERVICE_NOT_AVAILABLE: corresponding I-PDU group
                                     was stopped (or service failed due to development error)
                                 ->COM_BUSY: in case the TP-Buffer is locked for large data types
                                     handling
 Description:        The service Com_ReceiveSignalGroup copies the received signal group from the
                     I-PDU to the shadow buffer.
 *******************************************************************************************************************************/
uint8 Com_ReceiveSignalGroup(Com_SignalGroupIdType SignalGroupId)
{
    uint8 ComIPduIndex, ComSignalGroupIndex, ComGroupSignalIndex, BitIndex;
    Com_GroupSignalType* ComGroupSignalLocal;
    Com_IPduType* ComIPduLocal;

    /*[SWS_Com_00638] âŒˆThe service Com_ReceiveSignalGroup shall copy the received
    signal group from the I-PDU to the shadow buffer.âŒ‹ (SRS_Com_02041)*/
    if(SignalGroupId < ComMaxSignalGroupCnt)
    {
        /*After this call, the group signals could be copied from the shadow buffer to the RTE
        by calling Com_ReceiveSignal.*/
        /* Find the IPdu which contains this signal */

        ComGroupSignalLocal = &Com.ComConfig.ComGroupSignal[SignalGroupId];


        for(ComIPduIndex = 0; ComIPduIndex < ComMaxIPduCnt; ComIPduIndex++)
        {
            for(ComSignalGroupIndex = 0; Com.ComConfig.ComIPdu[ComIPduIndex].ComIPduSignalGroupRef[ComSignalGroupIndex] != NULL; ComSignalGroupIndex++)
            {
                if(Com.ComConfig.ComIPdu[ComIPduIndex].ComIPduSignalGroupRef[ComSignalGroupIndex]->ComHandleId == SignalGroupId)
                {
                    /* Get IPdu */
                    ComIPduLocal = &Com.ComConfig.ComIPdu[ComIPduIndex];

                    for(ComGroupSignalIndex = 0; ComIPduLocal->ComIPduSignalGroupRef[ComSignalGroupIndex]->ComGroupSignalRef[ComGroupSignalIndex] != NULL; ComGroupSignalIndex++)
                    {
                        /*Get Group Signal*/
                        ComGroupSignalLocal = ComIPduLocal->ComIPduSignalGroupRef[ComSignalGroupIndex]->ComGroupSignalRef[ComGroupSignalIndex];

                        /* Write data from IPdu to GroupSignal buffer to IPdu*/
                        for(BitIndex = ComGroupSignalLocal->ComBitPosition; BitIndex < ComGroupSignalLocal->ComBitPosition + ComGroupSignalLocal->ComBitSize; BitIndex++)
                        {
                            if((ComIPduLocal->ComBufferRef[BitIndex / 8] >> (BitIndex % 8)) & 1)
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
