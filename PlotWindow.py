import sys
 
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QSizePolicy, QSlider, QWidget, QLabel
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import pyqtgraph as pg

#import vispy.mpl_plot as plt

import random
import numpy as np

class PlotWindow(QMainWindow):
    def __init__(self, parent = None, roll_len = None, roll_len_aud = None, title="Signal Review"):
        super().__init__()
        self.setParent(parent)
        
        self.data = None
        self.dataAud = None
        self.roll_len = roll_len
        self.roll_len_aud = roll_len_aud
        
        self.setWindowTitle(title)
        self.setStyleSheet("background-color: #000000; color: #CCCCCC; border-color: #CCCCCC;")
        self.resize(800, 800)
        
        self.plotCanvas = []
        self.plots = []
        for i in range(0, 9):
            self.plotCanvas.append(pg.PlotWidget(parent = self, name = 'Plot1'))
            self.plots.append(self.plotCanvas[-1].plot())
            self.plots[-1].setPen((100,100,230))
            if i >= 7:
                self.plots[-1].setPen((230,100,100))
                self.plotCanvas[-1].setYRange(-(2**15), 2**15)
        
        self.channelSelLabel = QLabel("First EMG channel: 0")
        self.channelSel = QSlider(QtCore.Qt.Horizontal, self)
        
        self.channelSelLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.channelSelLabel.setMinimumSize(150, 20)
        
        layoutHorizontalChannelSel = QHBoxLayout()
        layoutHorizontalChannelSel.addWidget(self.channelSelLabel)
        layoutHorizontalChannelSel.addWidget(self.channelSel)
        
        mainLayout = QVBoxLayout()
        for i in range(0, 9):
            mainLayout.addWidget(self.plotCanvas[i])
            
        mainLayout.addLayout(layoutHorizontalChannelSel)
        
        mainWidget = QWidget()
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)
        
        self.channelSel.sliderReleased.connect(self.sliderReplot)
        self.channelSel.valueChanged.connect(self.displaySliderStatus)
        
        self.shortcutClose = QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self)
        self.shortcutClose.activated.connect(self.close)
        
    def displaySliderStatus(self, value):
        self.channelSelLabel.setText("First EMG channel: " + str(value))
    
    def roll(self, new_data, new_data_aud):
        if self.data is None:
            self.data = np.zeros((self.roll_len, new_data.shape[1]))
            self.dataAud = np.zeros((self.roll_len_aud, 2))
            self.channelSel.setMinimum(0)
            self.channelSel.setMaximum(new_data.shape[1] - 7)
            
        self.data[0:self.roll_len - len(new_data),:] = self.data[len(new_data):,:] 
        self.data[self.roll_len - len(new_data):,:] = new_data
    
        self.dataAud[0:self.roll_len_aud - len(new_data_aud),:] = self.dataAud[len(new_data_aud):,:] 
        self.dataAud[self.roll_len_aud - len(new_data_aud):,:] = new_data_aud
        
        self.replot(self.channelSel.value())
        
    def plot(self, data, dataAud):
        self.data = data
        self.dataAud = dataAud
        #self.channelSel.setValue(0)
        self.channelSel.setMinimum(0)
        self.channelSel.setMaximum(data.shape[1] - 7)
        self.replot(self.channelSel.value())
    
    def sliderReplot(self):
        self.replot(self.channelSel.value())
        
    def replot(self, firstChannel):
        if not self.data is None and self.isVisible():
            for i in range(0, 7):
                self.plots[i].setData(self.data[:,firstChannel + i])
            self.plots[7].setData(self.dataAud[:, 0])
            self.plots[8].setData(self.dataAud[:, 1])
        
#data = np.array([random.random() for i in range(2048*5*33)]).reshape(2048*5,33)
#dataAud = np.array([random.random() for i in range(16000*5*2)]).reshape(16000*5,2)
#app = QApplication(sys.argv)
#ex = PlotWindow()
#ex.show()
#ex.plot(data, dataAud)
#sys.exit(app.exec_())
