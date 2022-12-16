from Nodes import F0Calculator, AudioSource, Receiver, RunningNorm
import pyaudio
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from PlotWindow import PlotWindow

# recording - device name
DEVICE = "C510"

# display 
ROLL_LEN = 1000 # how many data points (10ms each) to show
MEDIAN_LEN = 5 # take median over how many past datapoints for smoothing?

# F0 extraction
F0_MIN = 50.0 # minimum f0 - set tightly for better quality of extraction
F0_MAX = 350.0 # maximum f0 - set tightly for better quality of extraction
HARMO_THRESH = 0.4 # threshold for when to consider something voiced vs unvoiced. needs to be tuned somewhat. higher -> usually noisier, lower -> might miss some speech
LP_CUTOFF_NORMALIZED = 0.9 # cutoff (0 to 1) for lowpass filter ran before extraction
PRE_GAIN_LINEAR = 10.0 # how much (constant, linear) gain to apply. 

if __name__ == '__main__':
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(p.get_device_info_by_index(i)["name"])
        
    window = None
    source = AudioSource.AudioSource(device_name_filter = DEVICE)
    norm = RunningNorm.RunningNorm()(source) # this is not a very good AGC but it is what I have
    f0Calc = F0Calculator.F0Calculator(128, 10, 16000, f0_min = F0_MIN, f0_max = F0_MAX, lp_cut = LP_CUTOFF_NORMALIZED, harmo_thresh = HARMO_THRESH, gain = PRE_GAIN_LINEAR)(norm)
    receiver = Receiver.Receiver()(f0Calc)
    source.start_processing()

    app = QApplication([])
    window = PlotWindow(roll_len = ROLL_LEN, maxval = F0_MAX)
    window.show()  

    median_data = []
    def roll_data():
        global median_data
        data = np.array(receiver.get_data(clear=True)).flatten()
        median_data.extend(list(data))
        median_data = median_data[-MEDIAN_LEN:]
        plot_data = np.array([np.median(median_data)] * len(data))
        if not len(plot_data) == 0:
            window.roll(plot_data.reshape(-1, 1))
    timer = QTimer()
    timer.timeout.connect(roll_data)
    timer.start(20)

    app.exec_()
    source.stop_processing()

