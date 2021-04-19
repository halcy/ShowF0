from multiprocessing import Process, Value, Lock

class RunManager():
    stop = Value('b', True)
    
    @staticmethod
    def beginRun():
        RunManager.stop.value = False
    
    @staticmethod
    def windDown():
        RunManager.stop.value = True
       
    @staticmethod
    def running():
        return not RunManager.stop.value
