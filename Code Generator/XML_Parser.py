import xml.etree.ElementTree as ET
tree = ET.parse('./XML_config.xml')
root = tree.getroot()

def signals_list():
    newsignals = []
    for child in root.iter('signal'):
        signals = {}
        for item in child:
            signals[item.tag] = item.text
        newsignals.append(signals)
    return newsignals

def GroupSignals_list():
    newgroupsignals = []
    for child in root.iter('GroupSignal'):
        groupsignals = {}
        for item in child:
            groupsignals[item.tag] = item.text
        newgroupsignals.append(groupsignals)
    return newgroupsignals

def SignalGroups_list():
    newsignalgroups = []
    for child in root.iter("SignalGroup"):
        signalgroup = {}
        for item in child:
            if item.tag == 'GroupSignalRefs':
                groupsignalrefs = []
                for refs in item:
                    groupsignalrefs.append(refs.text)
                signalgroup[item.tag] = groupsignalrefs
            else:
                signalgroup[item.tag] = item.text
        newsignalgroups.append(signalgroup)
    return newsignalgroups

def IPDUs_list():
    newgroupIPDUs = []
    for child in root.iter('IPDU'):
        groupIPDU = {}
        for item in child:
            if item.tag == 'IPduSignalGroupRefs':
                newsignalgroupRefs = []
                for g in item:
                   newsignalgroupRefs.append(g.text)
                groupIPDU[item.tag] = newsignalgroupRefs
            elif item.tag == 'IPduSignalRefs':
                newsignalRefs = []
                for s in item:
                   newsignalRefs.append(s.text) 
                groupIPDU[item.tag] = newsignalRefs
            elif item.tag == 'TxIPdu':
                txIPdu = {}
                for tx in item:
                    if tx.tag == 'TxModeFalse':
                        TxModeFalse = {}
                        for i in tx:
                            TxModeFalse[i.tag] = i.text
                        txIPdu[tx.tag] = TxModeFalse
                    else:
                        txIPdu[tx.tag] = tx.text 
                groupIPDU[item.tag] = txIPdu
            else:
                groupIPDU[item.tag] = item.text
        newgroupIPDUs.append(groupIPDU)
    return newgroupIPDUs


