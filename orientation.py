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



class Orientation_VectorLike(icetray.I3ConditionalModule):
    """
    Get PMT direction information
    """
    def __init__(self, ctx):
        super(Orientation_VectorLike,self).__init__(ctx)
        self.AddParameter("Pulses", "Name of pulses in the frame", "I3RecoPulseSeriesMapGen2")
        self.AddParameter('OnlyLC','Only LC pulses',False)
        self.AddParameter('KeyName', 'Key name for the results, default = SegmentationInfo_Recopulsesname_TimePercentileVal', None)
        self.AddParameter('TimePercentiles', 'Time percentile of hits to use in a list (so for all hits it would be [100]). Default [25,50,75,100]', [25,50,75,100])

    def Configure(self):
        self.pulses = self.GetParameter("Pulses")
        self.onlyLC = self.GetParameter("OnlyLC")
        key_name = self.GetParameter("KeyName")
        if key_name == None:
            self.key_name = "SegmentationInfo_"+self.pulses
        else:
            self.key_name = key_name
        self.timepercentiles = self.GetParameter("TimePercentiles")
        self.timeweight = False
        self.depthweight = False

    def Geometry(self, frame):
        #self.omgeo = frame['I3Geometry'].omgeo
        self.omgeomap = frame["I3OMGeoMap"]
        self.PMTdirections = []
        for k in self.omgeomap.keys():
            if k.string > 86:
                if self.omgeomap[k].omtype == 130:
                    for pmt in range(24):
                        thismodule = icetray.OMKey(k.string, k.om, pmt)
                        self.PMTdirections.append(self.omgeomap[thismodule].orientation.dir)
                    break
        self.PushFrame(frame)

    def Physics(self, frame):
        if len(self.PMTdirections) == 0:
            print "no mdom in this geometry"
            return True
        LC = dataclasses.I3RecoPulse.PulseFlags.LC
        source = frame[self.pulses]
        if type(source) != dataclasses.I3RecoPulseSeriesMap:
            source = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, self.pulses)
        maxtimes = self.filterhits_bytime(source = source,percentiles = self.timepercentiles)
        if maxtimes == False:
            i3map = dataclasses.I3MapStringDouble()
            for i in range(len(self.timepercentiles)):
                i3map["DirX_"+str(self.timepercentiles[i])+""] = -999999.
                i3map["DirY_"+str(self.timepercentiles[i])+""] = -999999.
                i3map["DirZ_"+str(self.timepercentiles[i])+""] = -999999.
                i3map["DirZenith_"+str(self.timepercentiles[i])+""] = -999999.
                i3map["DirAzimuth_"+str(self.timepercentiles[i])+""] = -999999.
                i3map["Module_"+str(self.timepercentiles[i])+""] = -999999.
                i3map["TotalCharge_"+str(self.timepercentiles[i])+""] = -999999.
            frame.Put(self.key_name,i3map)
            self.PushFrame(frame)
        else:
            counter = 0
            charges = []
            ydir_distribution = []
            zdir_distribution = []
            xdir_distribution = []
            if self.timeweight:
                times = []
            if self.depthweight:
                depths = []
            for i in range(len(maxtimes)):
                charges.append([])
                xdir_distribution.append([])
                ydir_distribution.append([])
                zdir_distribution.append([])
                if self.timeweight:
                    times.append([])
                if self.depthweight:
                    depths.append([])
            for key, pulses in source.items():
                if self.omgeomap[key].omtype == 130:
                    for pulse in pulses:
                        if self.onlyLC:
                            if pulse.flags != LC:
                                continue
                        for i in range(len(maxtimes)):
                            maxtime = maxtimes[i]
                            if pulse.time <= maxtime:
                                if self.timeweight:
                                    times[i].append(pulse.time)
                                if self.depthweight:
                                    depths[i].append(self.omgeomap[key].position.z)
                                charges[i].append(pulse.charge)
                                xdir_distribution[i].append(self.PMTdirections[key.pmt].x)
                                ydir_distribution[i].append(self.PMTdirections[key.pmt].y)
                                zdir_distribution[i].append(self.PMTdirections[key.pmt].z)
            i3map = dataclasses.I3MapStringDouble()
            for i in range(len(maxtimes)):
                if len(charges[i]) == 0:
                    i3map["DirX_"+str(self.timepercentiles[i])+""] = -999999.
                    i3map["DirY_"+str(self.timepercentiles[i])+""] = -999999.
                    i3map["DirZ_"+str(self.timepercentiles[i])+""] = -999999.
                    i3map["DirZenith_"+str(self.timepercentiles[i])+""] = -999999.
                    i3map["DirAzimuth_"+str(self.timepercentiles[i])+""] = -999999.
                    i3map["Module_"+str(self.timepercentiles[i])+""] = -999999.
                    i3map["TotalCharge_"+str(self.timepercentiles[i])+""] = -999999.
                else:
                    weights = numpy.array(charges[i])/sum(charges[i])
                    xdir = sum(numpy.array(xdir_distribution[i])*weights)
                    ydir = sum(numpy.array(ydir_distribution[i])*weights)
                    zdir = sum(numpy.array(zdir_distribution[i])*weights)
                    module = numpy.sqrt(xdir**2+ydir**2+zdir**2)
                    dir_i3vector = dataclasses.I3Direction(xdir/module,ydir/module,zdir/module)
                    zenith = dir_i3vector.zenith
                    azimuth = dir_i3vector.azimuth
                    i3map["DirX_"+str(self.timepercentiles[i])+""] = xdir/module
                    i3map["DirY_"+str(self.timepercentiles[i])+""] = ydir/module
                    i3map["DirZ_"+str(self.timepercentiles[i])+""] = zdir/module
                    i3map["DirZenith_"+str(self.timepercentiles[i])+""] = zenith
                    i3map["DirAzimuth_"+str(self.timepercentiles[i])+""] = azimuth
                    i3map["Module_"+str(self.timepercentiles[i])+""] = module
                    i3map["TotalCharge_"+str(self.timepercentiles[i])+""] = sum(charges[i])
            frame.Put(self.key_name,i3map)
            self.PushFrame(frame)
    
    def filterhits_bytime(self, source,percentiles):
        newpulses = dataclasses.I3RecoPulseSeriesMap
        LC = dataclasses.I3RecoPulse.PulseFlags.LC
        times = []
        for key, pulses in source.items():
            for pulse in pulses:
                if self.onlyLC:
                    if pulse.flags != LC:
                        continue
                times.append(pulse.time)
        if len(times) == 0:
            return False
        else:
            maxtimes = list()
            for i in range(len(percentiles)):
                maxtimes.append(numpy.percentile(times,percentiles[i]))
            return maxtimes



def testorientations():
    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', FilenameList=["/afs/ifh.de/user/c/clozano/lustre/Datasets/Phase1/GCDs/GeoCalibDetectorStatus_ICUpgrade.v47.mDOM.V3.i3.bz2",themainfolder+"nu_simulation/mdom/1230/genie_ICU_v47_mdom.1230.000000_output.i3.bz2"])
    #tray.AddModule('I3Reader', 'reader', FilenameList=["/afs/ifh.de/user/c/clozano/lustre/Datasets/Phase1/GCDs/GeoCalibDetectorStatus_ICUpgrade.v47.mDOM.V3.i3.bz2",themainfolder+"muongun/mDOM/muongun_ICU_v47_mDOM_47_step3.000_output.i3.bz2"])
    #tray.Add(Orientation, "Orientation",
    #         OnlyLC = False,
    #         KeyName = "Orientation",
    #         TimeWeight=True,
    #         DepthWeight=True)
    tray.Add(Orientation_VectorLike, "Orientation",
             OnlyLC = False,
             KeyName = "Orientation",
            )
    
    tray.Add('I3Writer',
        Filename          = "test.i3",
        Streams           = [icetray.I3Frame.DAQ,
                            icetray.I3Frame.Physics]
        )
    tray.Execute()
    tray.Finish()
    del tray
 


if __name__ == "__main__":
    testorientations()
