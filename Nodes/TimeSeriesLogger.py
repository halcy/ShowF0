import time
import os
import os.path
import numpy as np
import pprint

DO_TIMING = False
OUTPUT = False

class TimeSeriesLogger():
    series = {}
    eventCounters = {}
    allEventCounters = 0
    
    @staticmethod
    def logPoint(seriesName, dataPoint):
        if DO_TIMING:
            if seriesName not in TimeSeriesLogger.series:
                TimeSeriesLogger.series[seriesName] = []
                TimeSeriesLogger.eventCounters[seriesName] = 0
            
            logTime = time.time()
            
            dataPoint = np.array(dataPoint)
            if len(dataPoint.shape) == 1:
                dataPoint = np.array([dataPoint])
                
            for dataFrame in dataPoint:
                if isinstance(dataFrame[0], list) or (isinstance(dataFrame[0], np.ndarray) and len(dataFrame[0].shape) == 1):
                    dataFrame = np.array(dataFrame[0]).flatten()
                else:
                    dataFrame = np.array(dataFrame).flatten()
                pointToLog = [logTime, dataPoint.shape[0]]
                pointToLog.extend(list(dataFrame))
                pointToLog = np.array(pointToLog)
                TimeSeriesLogger.series[seriesName].append(pointToLog)
                TimeSeriesLogger.eventCounters[seriesName] += 1
                TimeSeriesLogger.allEventCounters += 1
            
            if OUTPUT == True and TimeSeriesLogger.allEventCounters > 10000:
                TimeSeriesLogger.allEventCounters = 0
                pprint.pprint(TimeSeriesLogger.eventCounters)
    
    
    @staticmethod
    def activateTiming(activate):
        global DO_TIMING
        DO_TIMING = activate
    
    @staticmethod
    def setBaseName(baseName):
        TimeSeriesLogger.baseName = baseName
    
    @staticmethod
    def writeLogs():
        if DO_TIMING:
            try:
                if not os.path.exists(TimeSeriesLogger.baseName):
                    os.makedirs(TimeSeriesLogger.baseName)
            except:
                pass
            
            for seriesName in TimeSeriesLogger.series:
                seriesArray = np.array(TimeSeriesLogger.series[seriesName])
                np.save(os.path.join(TimeSeriesLogger.baseName, seriesName + ".npy"), seriesArray)
