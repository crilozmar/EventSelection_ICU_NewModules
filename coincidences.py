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

class Coincify(icetray.I3ConditionalModule):
    """
    Apply a local coincidence condition to pulses
    """
    def __init__(self, ctx):
        super(Coincify,self).__init__(ctx)
        self.AddParameter("Pulses", "Name of pulses in the frame", "I3RecoPulseSeriesMapGen2")
        self.AddParameter("LCSpan", "number of DOM positions to look up and down the string", 1)
        self.AddParameter("LCWindow", "time window for coincidence on neighboring OMs", 1*icetray.I3Units.microsecond)
        self.AddParameter("ModuleWindow", "time window for coincidence on the same OM", 0.1*icetray.I3Units.microsecond)
        self.AddParameter("KeyName", "Key name with the main info about the coincidences", "CoincidencesInfo")
        self.AddParameter("Reset", "Reset any previous LC status to 0?", False)

    def Configure(self):
        self.pulses = self.GetParameter("Pulses")
        self.lc_span = self.GetParameter("LCSpan")
        self.lc_window = self.GetParameter("LCWindow")
        self.module_window = self.GetParameter("ModuleWindow")
        self.key_name = self.GetParameter("KeyName")
        self.reset = self.GetParameter("Reset")
    
    def Geometry(self, frame):
        omgeo = frame['I3Geometry'].omgeo
        self.n_pmts = max((k.pmt for k in omgeo.keys()))+1
        self.PushFrame(frame)
    
    def DAQ(self, frame):
        import numpy
        source = frame[self.pulses]
        
        if type(source) != dataclasses.I3RecoPulseSeriesMap:
                source = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, self.pulses)

        LC = dataclasses.I3RecoPulse.PulseFlags.LC #local coincidence flag?
        for key, pulses in source.items():
            # gather neighbors
            neighbors = []
            siblings = []
            for pmtnum in range(0, self.n_pmts):
                k = icetray.OMKey(key.string, key.om, pmtnum)
                if k != key and k in source:
                    siblings += [p.time for p in source[k]]
                for omnum in range(key.om-self.lc_span, key.om+self.lc_span+1):
                    if omnum < 1: continue
                    k = icetray.OMKey(key.string, omnum, pmtnum)
                    if k.om != key.om and k in source:
                        neighbors += [p.time for p in source[k]]
            neighbors = numpy.array(sorted(neighbors))
            siblings = numpy.array(sorted(siblings))
        
            times = numpy.array(sorted([p.time for p in pulses]))

            # count number of pulses on neighboring OMs in coincidence window
            n_neighbors = neighbors.searchsorted(times+self.lc_window) - neighbors.searchsorted(times-self.lc_window)
            # ditto for pulses on the same DOM
            n_siblings = siblings.searchsorted(times+self.module_window) - siblings.searchsorted(times-self.module_window)
            for i, keep in enumerate((n_neighbors > 0) | (n_siblings > 0)):
                pulse = pulses[i]
                if self.reset: pulse.flags = 0
                if keep:
                    pulse.flags |= LC
                else:
                    pulse.flags &= ~LC
                pulses[i] = pulse
            source[key] = pulses
        del frame[self.pulses]
        frame[self.pulses] = source
        
        total_charge = 0
        lc_charge = 0
        coincidences_counter = 0
        coincidences_modules = set()
        coincidences_pmts = set()
        for key, pulses in source.items():
            for p in pulses:
                total_charge += p.charge
                if (p.flags & LC) > 0:
                    coincidences_counter += 1
                    lc_charge += p.charge
                    coincidences_pmts.add(""+str(key.string)+","+str(key.om)+","+str(key.pmt)+"")
                    coincidences_modules.add(""+str(key.string)+","+str(key.om)+"")
        
        i3map = dataclasses.I3MapStringDouble()
        i3map["LCSpan"] = self.lc_span
        i3map["LCWindow"] = self.lc_window
        i3map["NumberOfCoincidences"] = coincidences_counter
        i3map["NumberCoincidencesPMTs"] = len(coincidences_pmts)
        i3map["NumberCoincidencesModules"] = len(coincidences_modules)
        i3map["LCCharge"] = lc_charge
        i3map["TotalCharge"] = total_charge
        frame.Put(self.key_name,i3map)
        self.PushFrame(frame)
        
        
class NewCoincidences(icetray.I3ConditionalModule):
    """
    Apply a local coincidence condition to pulses
    """
    def __init__(self, ctx):
        super(NewCoincidences,self).__init__(ctx)
        self.AddParameter("Pulses", "Name of pulses in the frame", "I3RecoPulseSeriesMapGen2")
        self.AddParameter("LCSpan", "number of DOM positions to look up and down the string", 1)
        self.AddParameter("LCWindow", "time window for coincidence on neighboring OMs", 1*icetray.I3Units.microsecond)
        self.AddParameter("ModuleWindow", "time window for coincidence on the same OM", 0.1*icetray.I3Units.microsecond)

    def Configure(self):
        self.pulses = self.GetParameter("Pulses")
        self.lc_span = self.GetParameter("LCSpan")
        self.lc_window = self.GetParameter("LCWindow")
        self.module_window = self.GetParameter("ModuleWindow")
    
    def Geometry(self, frame):
        self.omgeo = frame['I3Geometry'].omgeo
        self.n_pmts = max((k.pmt for k in self.omgeo.keys()))+1
        self.PMTdirections = []
        for k in self.omgeo.keys():
            if k.string > 86:
                if self.omgeo[k].omtype == 130:
                    for pmt in range(24):
                        thismodule = icetray.OMKey(k.string, k.om, pmt)
                        self.PMTdirections.append(self.omgeo[thismodule].orientation.dir)
                    break
        self.PushFrame(frame)
    
    def DAQ(self, frame):
        self.PushFrame(frame)

    def Physics(self, frame):
        import numpy
        import copy
        source = frame[self.pulses]
        if type(source) != dataclasses.I3RecoPulseSeriesMap:
                source = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, self.pulses)
        source_siblings = dataclasses.I3RecoPulseSeriesMap()
        source_neighbors = dataclasses.I3RecoPulseSeriesMap()
        #LC = dataclasses.I3RecoPulse.PulseFlags.LC #local coincidence flag?
        for key, pulses in source.items():
            # gather neighbors
            neighbors = []
            siblings = []
            for pmtnum in range(0, self.n_pmts):
                k = icetray.OMKey(key.string, key.om, pmtnum)
                if k != key and k in source:
                    siblings += [p.time for p in source[k]]
                for omnum in range(key.om-self.lc_span, key.om+self.lc_span+1):
                    if omnum < 1: continue
                    k = icetray.OMKey(key.string, omnum, pmtnum)
                    if k.om != key.om and k in source:
                        neighbors += [p.time for p in source[k]]
            neighbors = numpy.array(sorted(neighbors))
            siblings = numpy.array(sorted(siblings))
            times = numpy.array(sorted([p.time for p in pulses]))
            # count number of pulses on neighboring OMs in coincidence window
            n_neighbors = neighbors.searchsorted(times+self.lc_window) - neighbors.searchsorted(times-self.lc_window)
            # ditto for pulses on the same DOM
            n_siblings = siblings.searchsorted(times+self.module_window) - siblings.searchsorted(times-self.module_window)
            pulses_neighbors = dataclasses.vector_I3RecoPulse()
            for i, keep in enumerate(n_neighbors > 0):
                pulse = pulses[i]
                if keep:
                    pulses_neighbors.append(pulse)
            if len(pulses_neighbors) > 0:
                source_neighbors[key] = pulses_neighbors
                
            pulses_siblings = dataclasses.vector_I3RecoPulse()
            for i, keep in enumerate(n_siblings > 0):
                pulse = pulses[i]
                if keep:
                    pulses_siblings.append(pulse)
            if len(pulses_siblings) > 0:
                source_siblings[key] = pulses_siblings
        frame[self.pulses+"_IntraLC_"+str(self.module_window/icetray.I3Units.nanosecond)+"ns"] = source_siblings
        frame[self.pulses+"_InterLC_"+str(self.lc_window/icetray.I3Units.microsecond)+"ms"] = source_neighbors
        frame["IntraLC_info_"+self.pulses+"_"+str(self.module_window/icetray.I3Units.nanosecond)+"ns"] = self.WriteCoincidencesInfo(source_siblings, intracoincidences=True)
        frame["InterLC_info_"+self.pulses+"_"+str(self.lc_window/icetray.I3Units.microsecond)+"ms"] = self.WriteCoincidencesInfo(source_neighbors)
        self.PushFrame(frame)

    def WriteCoincidencesInfo(self, recopulses, intracoincidences=False):
        lc_charge = 0
        coincidences_counter = 0
        coincidences_modules = set()
        coincidences_pmts = set()
        depths = list()
        charges = list()
        if intracoincidences:
            pmts_hit = dict()
            pmts_hit_times = dict()
        for key, pulses in recopulses.items():
            for p in pulses:
                depths.append(self.omgeo[key].position.z)
                charges.append(p.charge)
                coincidences_counter += 1
                coincidences_pmts.add(""+str(key.string)+","+str(key.om)+","+str(key.pmt)+"")
                coincidences_modules.add(""+str(key.string)+","+str(key.om)+"")
                if intracoincidences:
                    if ""+str(key.string)+","+str(key.om)+"" not in pmts_hit:
                        pmts_hit[""+str(key.string)+","+str(key.om)+""] = [key.pmt]
                        pmts_hit_times[""+str(key.string)+","+str(key.om)+""] = [p.time]
                    else:
                        pmts_hit[""+str(key.string)+","+str(key.om)+""].append(key.pmt)
                        pmts_hit_times[""+str(key.string)+","+str(key.om)+""].append(p.time)
        if len(depths) > 0:
            meanz, z_var = self.weighted_avg_and_std(values=depths, weights=charges)
            totalcoverdepth = max(depths)-min(depths)
        else:
            meanz, z_var, totalcoverdepth = -999999., -999999., -999999.
        counters = [3,5,10,20] #how many modules are gonna be checked for the stats of "first modules blabla"
        if intracoincidences:
            if len(pmts_hit) > 0:
                opangle_list = list()
                meanopangle_list = list()
                maxopangle_list = list()
                repeated_coincidences_set = set() #how many times the same module saw coincidences
                repeated_coincidences_depth_set = set()
                for key in pmts_hit.keys():
                    om = int(key.split(",")[-1])
                    string = int(key.split(",")[0])
                    coinpmts_list = numpy.array(pmts_hit[key])
                    if len(coinpmts_list) < 1:
                        print "ERROR!! This should never happen"
                        continue      
                    cointimes_list = numpy.array(pmts_hit_times[key])
                    arr1inds = (-cointimes_list).argsort()
                    cointimes_list_sorted = cointimes_list[arr1inds[::-1]]
                    coinpmts_list_sorted = coinpmts_list[arr1inds[::-1]]
                    _opangles = list()
                    for i in range(len(coinpmts_list_sorted)-1):
                        for j in range(i+1,len(coinpmts_list_sorted)):
                            if abs(cointimes_list_sorted[j] - cointimes_list_sorted[i]) < abs(self.module_window):
                                _opangles.append(self.PMTdirections[coinpmts_list_sorted[i]].angle(self.PMTdirections[coinpmts_list_sorted[j]]) * 180.0 / numpy.pi)
                            else:
                                repeated_coincidences_set.add(""+str(string)+","+str(om)+"")
                                repeated_coincidences_depth_set.add(self.omgeo[icetray.OMKey(string, om, 0)].position.z)
                                break
                    opangle_list += _opangles
                    meanopangle_list.append(numpy.average(_opangles))
                    maxopangle_list.append(max(_opangles))
                repeated_coincidences = len(repeated_coincidences_set)
                meandepth_firstcoins = dict() #this will store the mean depths of the first OMs with intra coincidences
                vardepth_firstcoins = dict() #ditto for the variance
                firstcoins_depth = dict()
                mintime_dict = {k: min(v) for k, v in pmts_hit_times.items()} 
                for key, value in sorted(mintime_dict.iteritems(), key=lambda (k,v): (v,k)):
                    om = int(key.split(",")[-1])
                    string = int(key.split(",")[0])
                    boolcontrol = False
                    for i in counters:
                        try:
                            if i > len(firstcoins_depth[i]):
                                firstcoins_depth[i].append(self.omgeo[icetray.OMKey(string, om, 0)].position.z)
                                boolcontrol = True
                        except:
                            firstcoins_depth[i] = [self.omgeo[icetray.OMKey(string, om, 0)].position.z]
                            boolcontrol = True
                    if not boolcontrol:
                        break
                for key, values in firstcoins_depth.iteritems():
                    mean, var = self.weighted_avg_and_std(values=values,weights=numpy.ones_like(values)) #no weights
                    meandepth_firstcoins[key] = mean
                    vardepth_firstcoins[key] = var
                meanopangle_total, varopangle_total = self.weighted_avg_and_std(values=_opangles,weights=numpy.ones_like(_opangles)) #no weights
                meanopangle, varopangle = self.weighted_avg_and_std(values=meanopangle_list,weights=numpy.ones_like(meanopangle_list)) #no weights
                mean_maxopangle, var_maxopangle = self.weighted_avg_and_std(values=maxopangle_list,weights=numpy.ones_like(maxopangle_list)) #no weights
                totalmax = max(maxopangle_list)
                if repeated_coincidences > 0:
                    repeated_coincidences_depth = max(list(repeated_coincidences_depth_set))-min(list(repeated_coincidences_depth_set))
                else:
                    repeated_coincidences_depth = 0.
            else:
                meanopangle_total = -999999.
                varopangle_total = -999999.
                meanopangle = -999999.
                varopangle = -999999.
                mean_maxopangle = -999999.
                var_maxopangle = -999999.
                totalmax = -999999.
                repeated_coincidences = -999999.
                repeated_coincidences_depth = -999999.
                meandepth_firstcoins = dict()
                vardepth_firstcoins = dict()
                for i in counters:
                    meandepth_firstcoins[i] = -999999.
                    vardepth_firstcoins[i] = -999999.
        lc_charge += sum(charges)
        i3map = dataclasses.I3MapStringDouble()
        i3map["NumberOfCoincidences"] = coincidences_counter
        i3map["NumberCoincidencesPMTs"] = len(coincidences_pmts)
        i3map["NumberCoincidencesModules"] = len(coincidences_modules)
        i3map["LCCharge"] = lc_charge
        i3map["MeanZ"] = meanz
        i3map["Z_var"] = z_var
        i3map["TotalDepth"] = totalcoverdepth
        if intracoincidences: #Op angle: angle between PMTs in intracoincidences
            i3map["OpAngle_TotalMean"] = meanopangle_total #Mean of all op angles
            i3map["OpAngle_TotalVar"] = varopangle_total
            i3map["OpAngle_Means"] = meanopangle #Mean of each OM mean op angle
            i3map["OpAngle_Vars"] = varopangle
            i3map["OpAngle_MeanMax"] = mean_maxopangle #Mean of max opening angles
            i3map["OpAngle_VarMax"] = var_maxopangle
            i3map["OpAngle_Max"] = totalmax #Max opening angle
            i3map["RepeatedOMs"] = repeated_coincidences #OMs which have intracoincidences more than once
            i3map["RepeatedOMs_depth"] = repeated_coincidences_depth #Depth cover by the repeated DOMs
            for key in meandepth_firstcoins.keys():
                i3map["First"+str(key)+"Coincidences_MeanZ"] = meandepth_firstcoins[key]
                i3map["First"+str(key)+"Coincidences_Z_var"] = vardepth_firstcoins[key]
        return i3map
    
    def weighted_avg_and_std(self,values, weights):
        average = numpy.average(values, weights=weights)
        variance = numpy.average((values-average)**2, weights=weights)
        return average, numpy.sqrt(variance)





def testCoincify():
    recopulses = "I3RecoPulseSeriesMapGen2"
    keep_keys = ['I3MCTree', 'I3EventHeader']
    LCWindow = [1*icetray.I3Units.microsecond,2*icetray.I3Units.microsecond]
    ModuleWindow = [100.*icetray.I3Units.ns, 20.*icetray.I3Units.ns]
    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', FilenameList=["/afs/ifh.de/user/c/clozano/lustre/Datasets/Phase1/GCDs/GeoCalibDetectorStatus_ICUpgrade.v53.mixed.V0.i3.bz2","/afs/ifh.de/user/c/clozano/lustre/Datasets/Phase1/nu_simulation/mixed/1251/genie_ICU_v53_mixed.1251.000142.i3.gz"])
    for i in range(len(LCWindow)):
        tray.Add(NewCoincidences, "testcoincify_"+str(i),
                 LCWindow = LCWindow[i],
                 ModuleWindow = ModuleWindow[i])
        keep_keys.append(recopulses+"_InterLC_"+str(LCWindow[i]/icetray.I3Units.microsecond)+"ms")
        keep_keys.append(recopulses+"_IntraLC_"+str(ModuleWindow[i]/icetray.I3Units.ns)+"ns")
        keep_keys.append("InterLC_info_"+recopulses+"_"+str(LCWindow[i]/icetray.I3Units.microsecond)+"ms")
        keep_keys.append("IntraLC_info_"+recopulses+"_"+str(ModuleWindow[i]/icetray.I3Units.ns)+"ns")
    print keep_keys
    tray.Add('Keep', 'keeper', keys=keep_keys)
    tray.Add('I3Writer',
        Filename          = "test.i3",
        Streams           = [icetray.I3Frame.DAQ,
                            icetray.I3Frame.Physics]
        )
    tray.Execute(200)
    tray.Finish()
    del tray


if __name__ == "__main__":
    testCoincify()
