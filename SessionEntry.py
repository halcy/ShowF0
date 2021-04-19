# -*- coding: utf-8 -*-

import sys
import os
import glob

from PyQt5 import QtCore, QtGui, QtWidgets

import RecordData

class SessionEntryWindow(QtWidgets.QMainWindow):

    def __init__(self, *args):
        super(SessionEntryWindow, self).__init__()
        
        self.settings = QtCore.QSettings("SilentEMGRecorder")
        self.createComponents()
        self.createLayout()
        self.connectEvents()
        self.title = "Silent-EMG Recorder"
        
        self.setWindowTitle(self.title)
        self.resize(1400, 400)
        self.setStyleSheet('font-size: 16pt;')
        
        self.recordingWindow = None
    
    def createComponents(self):
        self.labelBasePath = QtWidgets.QLabel("Base path")
        self.labelSessionName = QtWidgets.QLabel("Session name")
        self.labelSpeakerName = QtWidgets.QLabel("Speaker name")
        self.labelPromptList = QtWidgets.QLabel("Prompt list")
        self.labelAcqConfig = QtWidgets.QLabel("Recording setup")
        self.labelParallelPath = QtWidgets.QLabel("Speak-along audio path")
        
        self.inputBasePath = QtWidgets.QLineEdit()
        self.inputBasePath.setStyleSheet('font-size: 18pt;')
        
        self.inputSessionName = QtWidgets.QLineEdit()
        self.inputSessionName.setStyleSheet('font-size: 18pt;')
        
        self.inputSpeakerName = QtWidgets.QLineEdit()
        self.inputSpeakerName.setStyleSheet('font-size: 18pt;')
        
        self.inputParallelPath = QtWidgets.QLineEdit()
        self.inputParallelPath.setStyleSheet('font-size: 18pt;')
        
        
        self.buttonBasePathSel = QtWidgets.QPushButton("...")
        self.buttonParallelPathSel = QtWidgets.QPushButton("...")
        self.buttonEnter = QtWidgets.QPushButton("Start / resume recording session")
        
        self.comboPromptList =  QtWidgets.QComboBox()
        self.comboPromptList.setStyleSheet('font-size: 18pt;')
        
        self.comboAcqConfig =  QtWidgets.QComboBox()
        self.comboAcqConfig.setStyleSheet('font-size: 18pt;')
        
        self.inputBasePath.setText(self.settings.value("BasePath"))
        
        promptFileDir = os.path.join(os.path.dirname(__file__), 'PromptLists')
        for promptFile in glob.glob(promptFileDir + "/*.txt"):
            promptFileProper = os.path.basename(promptFile)[:-4]
            self.comboPromptList.addItem(promptFileProper)
        
        acqConfigDir = os.path.join(os.path.dirname(__file__), 'AcquisitionConfigs')
        for acqConfigFile in glob.glob(acqConfigDir + "/*.json"):
            acqConfigFileProper = os.path.basename(acqConfigFile)[:-5]
            self.comboAcqConfig.addItem(acqConfigFileProper)
        
    def connectEvents(self):
        self.buttonEnter.clicked.connect(self.beginRecordingSession)
        self.buttonBasePathSel.clicked.connect(self.selectBasePath)
        self.buttonParallelPathSel.clicked.connect(self.selectParallelPath)
        
    def createLayout(self): 
        mainLayout = QtWidgets.QVBoxLayout()
        layoutHorizontalBasePath = QtWidgets.QHBoxLayout()
        layoutHorizontalSpeaker = QtWidgets.QHBoxLayout()        
        layoutHorizontalSession = QtWidgets.QHBoxLayout()
        layoutHorizontalPromptList = QtWidgets.QHBoxLayout()
        layoutHorizontalAcqConfig = QtWidgets.QHBoxLayout()
        layoutHorizontalParallelPath = QtWidgets.QHBoxLayout()
        layoutHorizontalEnter = QtWidgets.QHBoxLayout()
        
        self.inputBasePath.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.inputSpeakerName.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.inputSessionName.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.inputParallelPath.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.comboPromptList.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.comboAcqConfig.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        
        self.buttonBasePathSel.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.buttonParallelPathSel.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.buttonEnter.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        
        self.labelBasePath.setMinimumSize(180, 30)
        self.labelSpeakerName.setMinimumSize(180, 30)        
        self.labelSessionName.setMinimumSize(180, 30)
        self.labelPromptList.setMinimumSize(180, 30)
        self.labelAcqConfig.setMinimumSize(180, 30)
        
        self.inputBasePath.setMinimumSize(750, 30)
        self.inputSessionName.setMinimumSize(750, 30)
        self.inputSpeakerName.setMinimumSize(750, 30)
        self.inputParallelPath.setMinimumSize(750, 30)
        
        self.buttonBasePathSel.setMinimumSize(30, 30)
        self.buttonParallelPathSel.setMinimumSize(30, 30)
        self.buttonEnter.setMinimumSize(550, 30)
        
        self.comboPromptList.setMinimumSize(750, 30)
        self.comboAcqConfig.setMinimumSize(750, 30)
        
        layoutHorizontalBasePath.addWidget(self.labelBasePath)
        layoutHorizontalBasePath.addWidget(self.inputBasePath)
        layoutHorizontalBasePath.addWidget(self.buttonBasePathSel)
        
        layoutHorizontalSession.addWidget(self.labelSessionName)
        layoutHorizontalSession.addWidget(self.inputSessionName)
        
        layoutHorizontalSpeaker.addWidget(self.labelSpeakerName)
        layoutHorizontalSpeaker.addWidget(self.inputSpeakerName)
        
        layoutHorizontalPromptList.addWidget(self.labelPromptList)
        layoutHorizontalPromptList.addWidget(self.comboPromptList)
        
        layoutHorizontalAcqConfig.addWidget(self.labelAcqConfig)
        layoutHorizontalAcqConfig.addWidget(self.comboAcqConfig)
        
        layoutHorizontalParallelPath.addWidget(self.labelParallelPath)
        layoutHorizontalParallelPath.addWidget(self.inputParallelPath)
        layoutHorizontalParallelPath.addWidget(self.buttonParallelPathSel)
        
        layoutHorizontalEnter.addWidget(self.buttonEnter)
        
        mainLayout.addLayout(layoutHorizontalBasePath)
        mainLayout.addLayout(layoutHorizontalSpeaker)
        mainLayout.addLayout(layoutHorizontalSession) 
        mainLayout.addLayout(layoutHorizontalPromptList)
        mainLayout.addLayout(layoutHorizontalAcqConfig)
        mainLayout.addLayout(layoutHorizontalParallelPath)
        mainLayout.addLayout(layoutHorizontalEnter)
        
        mainWidget = QtWidgets.QWidget()
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)
    
    def selectBasePath(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        self.inputBasePath.setText(path)
        self.settings.setValue("BasePath", path)
        
    def selectParallelPath(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        self.inputParallelPath.setText(path)
    
    
    def beginRecordingSession(self):
        parallelPath = None
        if self.inputParallelPath.text() and not self.inputParallelPath.text() is None:
            parallelPath = self.inputParallelPath.text().strip()
        
        if (self.inputSessionName.text()) and (self.inputSpeakerName.text()):
            self.recordingWindow = RecordData.RecordDataWindow(
                self.inputBasePath.text(),
                self.inputSpeakerName.text(), 
                self.inputSessionName.text(),
                os.path.join(os.path.dirname(__file__), 'PromptLists', self.comboPromptList.currentText() + ".txt"),
                os.path.join(os.path.dirname(__file__), 'AcquisitionConfigs', self.comboAcqConfig.currentText() + ".json"),
                parallelPath = parallelPath,
            )
            self.recordingWindow.show()
            self.hide()
        else:
            pass
            
