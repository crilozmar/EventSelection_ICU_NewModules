#!/afs/ifh.de/user/c/clozano/lustre/virtualenvs/test/bin/python
#METAPROJECT /afs/ifh.de/user/c/clozano/lustre/Icetray/combo_stable_svn/build/
import os, sys, numpy, matplotlib
import tables, pandas
matplotlib.use('Agg')
from matplotlib import pyplot
import copy
import pybdt
from pybdt import ml, util
import time
from datetime import timedelta
from sklearn.model_selection import train_test_split
from pprint import pprint
from icecube import icetray, dataclasses, dataio
from I3Tray import I3Tray
from icecube.icetray import OMKey
from icecube.hdfwriter import I3HDFWriter
from icecube.icetray import I3Units

class MyVeto(icetray.I3ConditionalModule):
    """
    Get simple VetoPulses
    """
    def __init__(self, ctx):
        super(MyVeto,self).__init__(ctx)
        self.AddParameter("Pulses", "Name of pulses in the frame", "I3RecoPulseSeriesMapGen2")
        self.AddParameter("nRows", "Number of rows to use like a Veto - Will only use upper part of mDOMs or d-Eggs" , 2)

    def Configure(self):
        self.pulses = self.GetParameter("Pulses")
        self.nRows = self.GetParameter("nRows")
    
    def Geometry(self, frame):
        self.subdetectors = frame["Subdetectors"]
        self.omgeomap = frame["I3OMGeoMap"]
        self.PushFrame(frame)

    def DAQ(self, frame):
        self.PushFrame(frame)
 
    def Physics(self,frame):
        source = frame[self.pulses]
        if type(source) != dataclasses.I3RecoPulseSeriesMap:
            source = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, self.pulses)
        vetopulses = dataclasses.I3RecoPulseSeriesMap()
        for key, pulses in source.iteritems():
            if key.string > 86 and key.om <= self.nRows: #key.om starts in 1
                if self.omgeomap[key].omtype == 120: #degg
                    if key.pmt == 0:
                        vetopulses[key] = pulses
                elif self.omgeomap[key].omtype == 130: #mdom
                    if key.pmt < 12: # from 0 to 11
                        vetopulses[key] = pulses
        nvetohits = 0
        chargevetohits = 0.
        ndiffmodules_vetohits_set = set()
        for key, pulses in vetopulses.iteritems():
            for pulse in pulses:
                nvetohits += 1
                chargevetohits += pulse.charge
            ndiffmodules_vetohits_set.add(""+str(key.string)+","+str(key.om))
        ndiffmodules_vetohits = len(ndiffmodules_vetohits_set)
        ndiffpmts_vetohits = len(vetopulses)
        #now we write the stuff
        frame.Put("MyVetoHits_"+str(self.nRows)+"rows_"+self.pulses,vetopulses)
        i3map = dataclasses.I3MapStringDouble()
        i3map["nVetoHits"] = nvetohits
        i3map["chargeVetoHits"] = chargevetohits
        i3map["nModules_VetoHits"] = ndiffmodules_vetohits
        i3map["nPMTs_VetoHits"] = ndiffpmts_vetohits
        frame.Put("InfoMyVetoHits_"+str(self.nRows)+"rows_"+self.pulses,i3map)
        self.PushFrame(frame)




def testVeto():
    recopulses = "I3RecoPulseSeriesMapGen2"
    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', FilenameList=["/afs/ifh.de/user/c/clozano/lustre/Datasets/Phase1/GCDs/GeoCalibDetectorStatus_ICUpgrade.v53.mixed.V0.i3.bz2","/afs/ifh.de/user/c/clozano/lustre/Datasets/Phase1/results/dataset_with_cuts_15_02_19/nu_simulation/mixed/1251/genie_ICU_v53_mixed.1251.000042_output.i3.bz2"])
    tray.Add(MyVeto, "testingveto",
                 Pulses = recopulses,
                 nRows = 2)
    tray.Add('I3Writer',
        Filename          = "test.i3",
        Streams           = [icetray.I3Frame.DAQ,
                            icetray.I3Frame.Physics]
        )
    tray.Execute(200)
    tray.Finish()
    del tray


if __name__ == "__main__":
    testVeto()
