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



def check_volume(frame):
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    # "Check whether the interaction vertex is inside the instrumented volume for the case of neutrinos" #
    # "Check whether the muon or its secondaries go inside the instrumented volume anytime 
    #
    #Geo coordinates must be sorted to close the area in x,y. String in the middle is therefore not included
    #v47
    #z_min = -450
    #z_max = -200
    #x_coord = numpy.array([-5.55,9.63,26.22,32.43,8.14,-14.30])
    #y_coord = numpy.array([-20.96,2.11,-12.59,-54.81,-69.23,-45.57])
    #v53
    z_min = -480
    z_max = -200
    x_coord = numpy.array([26.96, 62.62, 89.29, 57.29, 14.29, 18.29])
    y_coord = numpy.array([-31.19, -35.16,-59.00, -83.69, -80.56,-51.05])
    x_y = numpy.column_stack((x_coord,y_coord))
    polygon = Polygon(x_y)
    tree = frame["I3MCTree"]
    truth = frame["I3MCTree"][0]
    truth_type = str(truth.type)
    if truth_type == "unknown" or truth_type == "MuMinus" or truth_type == "MuPlus":
        for particle in tree:
            x = particle.pos.x/I3Units.m
            y = particle.pos.y/I3Units.m
            z = particle.pos.z/I3Units.m
            point = Point(x,y)
            if z > z_min and z< z_max and polygon.contains(point):
                frame.Put('MyContainmentFilter', icetray.I3Bool(True))
                return True
        frame.Put('MyContainmentFilter', icetray.I3Bool(False))
        return False
    else:
        x = truth.pos.x/I3Units.m
        y = truth.pos.y/I3Units.m
        z = truth.pos.z/I3Units.m
        point = Point(x,y)
        if z > z_min and z< z_max and polygon.contains(point):
            frame.Put('MyContainmentFilter', icetray.I3Bool(True))
            return True
        else:
            frame.Put('MyContainmentFilter', icetray.I3Bool(False))
            return False

def testCheckVolume():
    print "*************** Testing neutrinos *******************"
    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', FilenameList=["/afs/ifh.de/user/c/clozano/lustre/Datasets/Phase1/GCDs/GeoCalibDetectorStatus_ICUpgrade.v53.mixed.V0.i3.bz2","/afs/ifh.de/user/c/clozano/lustre/Datasets/Phase1/results/dataset_with_cuts_15_02_19/nu_simulation/mixed/1251/genie_ICU_v53_mixed.1251.000042_output.i3.bz2"])
    tray.Add(check_volume, "testcoincify", Streams = [icetray.I3Frame.DAQ,icetray.I3Frame.Physics])
    tray.Add('I3Writer',
        Filename          = "test_nu.i3",
        Streams           = [icetray.I3Frame.DAQ,
                            icetray.I3Frame.Physics]
        )
    tray.Execute(100)
    tray.Finish()
    del tray
    print "*************** Testing muons *******************"
    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', FilenameList=["/afs/ifh.de/user/c/clozano/lustre/Datasets/Phase1/GCDs/GeoCalibDetectorStatus_ICUpgrade.v53.mixed.V0.i3.bz2", "/afs/ifh.de/user/c/clozano/lustre/Datasets/Phase1/results/dataset_with_cuts_15_02_19/muongun/mixed/MuonGun_ICUpgrade.v53.mixed.001351.000985_output.i3.bz2"])
    tray.Add(check_volume, "testcoincify", Streams = [icetray.I3Frame.DAQ,icetray.I3Frame.Physics])
    tray.Add('I3Writer',
        Filename          = "test_muons.i3",
        Streams           = [icetray.I3Frame.DAQ,
                            icetray.I3Frame.Physics]
        )
    tray.Execute(100)
    tray.Finish()
    del tray


if __name__ == "__main__":
    testCheckVolume()
