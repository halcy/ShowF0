import sys
 
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QSizePolicy, QSlider, QWidget, QLabel
from PyQt5.QtGui import QIcon

import pyqtgraph as pg

#import vispy.mpl_plot as plt

import random
import numpy as np

class PlotWindow(QMainWindow):
    def __init__(self, parent = None, roll_len = None, maxval = None, title="Signal Review"):
        super().__init__()
        self.setParent(parent)
        
        self.data = None
        self.dataAud = None
        self.roll_len = roll_len
        
        self.setWindowTitle(title)
        self.setStyleSheet("background-color: #000000; color: #CCCCCC; border-color: #CCCCCC;")
        self.resize(1000, 300)
        
        self.plotCanvas = []
        self.plots = []

        self.plotCanvas.append(pg.PlotWidget(parent = self, name = 'Plot1'))
        self.plots.append(self.plotCanvas[-1].plot())
        self.plots[-1].setPen((100,100,230))
        self.plotCanvas[-1].setYRange(0, maxval)
        self.plotCanvas[-1].showGrid(x = True, y = True, alpha = 0.3)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.plotCanvas[-1])
                
        mainWidget = QWidget()
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)
        
        self.shortcutClose = QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self)
        self.shortcutClose.activated.connect(self.close)
       
    def roll(self, new_data):
        if self.data is None:
            self.data = np.zeros((self.roll_len, new_data.shape[1]))
            
        self.data[0:self.roll_len - len(new_data),:] = self.data[len(new_data):,:] 
        self.data[self.roll_len - len(new_data):,:] = new_data
    
        self.replot(0)
        
    def plot(self, data):
        self.data = data
        
    def replot(self, firstChannel):
        if not self.data is None and self.isVisible():
            self.plots[0].setData(self.data[:,firstChannel + 0])
        
#data = np.array([random.random() for i in range(2048*5*33)]).reshape(2048*5,33)
#dataAud = np.array([random.random() for i in range(16000*5*2)]).reshape(16000*5,2)
#app = QApplication(sys.argv)
#ex = PlotWindow()
#ex.show()
#ex.plot(data, dataAud)
#sys.exit(app.exec_())
