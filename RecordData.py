#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from random import randint
import random
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from os.path import expanduser
import time
import subprocess

import RecordingManager
import PlotWindow

glyphs = {
    "to_record": "○",
    "recording": "⭗",
    "recorded": "●",
}

class RecordDataWindow(QtWidgets.QMainWindow):
    def __init__(self, basePath, speaker, session, uttList, recordingSetup, parallelPath = None, feedbackPath = None, preproFeaturesPath = None):
        print("Openened recording window to record " + speaker + "_" + session + " to " + basePath + " from list " + uttList)
        super(RecordDataWindow, self).__init__()

        # Create recording manager
        self.dummyRecord = False
        if speaker.lower().startswith("dummy") or session.lower().startswith("dummy"):
            self.dummyRecord = True
            self.recordingManager = RecordingManager.RecordingManager(basePath, speaker, session, uttList, recordingSetup, dummyRecord = True, parallelPath = parallelPath)
        else:
            self.recordingManager = RecordingManager.RecordingManager(basePath, speaker, session, uttList, recordingSetup, parallelPath = parallelPath)

        # Set up window properties
        self.title = "Silent-EMG Recorder - Speaker '" + speaker + "', Session '" + session + "'"
        self.setWindowTitle(self.title)
        self.resize(800, 200)
        self.setStyleSheet('font-size: 18pt;')

        # Create a plot window
        self.plotWindow = PlotWindow.PlotWindow()
        self.plotWindowRunning = PlotWindow.PlotWindow(roll_len = 2048 * 4, roll_len_aud = 16000 * 4, title = "Live View") # TODO do this right

        # Create and link up GUI components
        self.createMenuBar()
        self.createComponents()
        self.createLayout()
        self.connectEvents()
        
        self.lastUttEMGFile = None
        self.lastUttAudioFile = None
    
        self.grabKeyboard()
        
        self.plotTimer = QtCore.QTimer()
        self.plotTimer.timeout.connect(self.liveReplot)
        self.plotTimer.start(100)
    
    def liveReplot(self):
        if not self.dummyRecord:
            try:
                emg_data, audio_data = self.recordingManager.get_newest()
                self.plotWindowRunning.roll(emg_data, audio_data)
            except:
                pass
    
    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Space and e.isAutoRepeat() == False:
            self.buttonRecord.setDown(True)
            self.startRecording()
            e.accept()
        super(RecordDataWindow, self).keyReleaseEvent(e)
        
    def keyReleaseEvent(self, e):
        if e.key() == QtCore.Qt.Key_Space and e.isAutoRepeat() == False:
            self.buttonRecord.setDown(False)
            self.stopRecording()
            e.accept()
        else:
            e.ignore()
        super(RecordDataWindow, self).keyReleaseEvent(e)
        
    def closeEvent(self, event):
        self.plotWindow.close()
        self.plotWindowRunning.close()
    
    def createMenuBar(self):
        menuTools = self.menuBar().addMenu("Tools")
        
        self.actionWriteTranscript = QtWidgets.QAction("Write transcript", self)
        self.actionShowPlot = QtWidgets.QAction("Signal review", self)
        self.actionShowPlotRunning = QtWidgets.QAction("Live view", self)
        self.actionExit = QtWidgets.QAction("Exit", self)
        
        menuTools.addAction(self.actionWriteTranscript)
        menuTools.addAction(self.actionShowPlot)
        menuTools.addAction(self.actionShowPlotRunning)
        menuTools.addAction(self.actionExit)
        
    def createComponents(self):
        """
        Create window components
        """
        self.buttonRecord = QtWidgets.QPushButton("Record (hold)")
        self.buttonRecord.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.labelPrompt = QtWidgets.QLabel()
        self.labelPrompt.setStyleSheet('font-size: 24pt; border: 5px solid black; background: #DDDDDD; color: #000000;')
        self.labelPrompt.setAlignment(QtCore.Qt.AlignCenter)
        self.labelPrompt.setWordWrap(True)
        self.labelPrompt.setText("Wir müssen das Problem beraten und ich hoffe auch lösen, denn wenn wir es nicht lösen, bevor diese königinnen nester gebaut und kolonien gegründet haben, werden zehntausende von ameisenköniginnen dem menschen die vorherschaft über die erde nehmen und ihn vernichten!")
        
        self.labelStatus = QtWidgets.QLabel()
        self.labelStatus.setStyleSheet('font-size: 24pt; background: rgba(0.0, 0.0, 0.0, 0.0); color: #000000; margin: -12px 1px 0px 0px;')
        self.labelStatus.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        self.setStatusIcon("to_record")
        
        self.buttonPrev = QtWidgets.QPushButton("Go back one")
        self.buttonPlay = QtWidgets.QPushButton("Play last recorded")        
        self.buttonNext = QtWidgets.QPushButton("Skip to next")

        self.statusBar().setStyleSheet("QStatusBar { border-top: 1px solid rgba(0.0, 0.0, 0.0, 0.2); }")
        self.displayReadyStatus()
        
        self.updateGUI()
        
    def updateGUI(self):
        prompt = self.recordingManager.getCurrentPrompt()
        
        haveHilite = False
        inHilite = False
        promptNew = ""
        for promptChar in prompt:
            if promptChar == "_":
                haveHilite = True
                if inHilite == False:
                    promptNew += '<span style="color:#FF0000;">'
                    inHilite = True
                else:
                    promptNew += "</span>"
                    inHilite = False
            else:
                promptNew += promptChar
                
        if haveHilite:
            promptNew = '<span style="color:#777777;">' + promptNew + '</span>'
        
        self.labelPrompt.setText(promptNew)
        self.labelStatus.setText(glyphs["to_record"])
        
        uttFiles = self.recordingManager.getCurrentUtteranceFiles()
        if not uttFiles is None:
            self.buttonPlay.setText("Play utterance")
            self.setStatusIcon("recorded")
        else:
            self.buttonPlay.setText("Play last recorded")
            self.setStatusIcon("to_record")
            
    def displayReadyStatus(self):
        promptPos, promptCount = self.recordingManager.getPromptPosCount()
        self.statusBar().showMessage("Ready - Utterance {0:04d} of {1:04d}".format(promptPos + 1, promptCount))
      
    def setStatusIcon(self, iconType):
        self.labelStatus.setText(glyphs[iconType])

    def createLayout(self):
        """
        Layout window components
        """
        mainLayout = QtWidgets.QVBoxLayout()
        layoutHorizontal = QtWidgets.QHBoxLayout()
        stackLayout = QtWidgets.QStackedLayout()
        stackLayout.setStackingMode(1)
        
        self.labelPrompt.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.labelStatus.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        
        self.buttonRecord.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.buttonPrev.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.buttonPlay.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.buttonNext.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        
        self.labelPrompt.setMinimumSize(600, 300)
        self.labelStatus.setFixedHeight(40)
        
        self.buttonRecord.setMinimumSize(250, 30)
        self.buttonPrev.setMinimumSize(250, 30)
        self.buttonPlay.setMinimumSize(250, 30)
        self.buttonNext.setMinimumSize(250, 30)

        stackLayout.addWidget(self.labelPrompt)
        stackLayout.addWidget(self.labelStatus)
        mainLayout.addLayout(stackLayout)
        mainLayout.addWidget(self.buttonRecord)

        layoutHorizontal.addWidget(self.buttonPrev)        
        layoutHorizontal.addWidget(self.buttonPlay)
        layoutHorizontal.addWidget(self.buttonNext)
        
        mainLayout.addLayout(layoutHorizontal)
        
        widgetZentral = QtWidgets.QWidget()
        widgetZentral.setLayout(mainLayout)
        self.setCentralWidget(widgetZentral)

    def connectEvents(self):
        """
        Connect UI actions to functionality
        """
        self.buttonRecord.pressed.connect(self.startRecording)
        self.buttonRecord.released.connect(self.stopRecording)
        self.buttonNext.clicked.connect(self.nextUtterance)
        self.buttonPrev.clicked.connect(self.prevUtterance)
        self.buttonPlay.clicked.connect(self.playAudio)
        self.actionWriteTranscript.triggered.connect(self.writeTranscript)
        self.actionShowPlot.triggered.connect(self.showPlotWindow)
        self.actionShowPlotRunning.triggered.connect(self.showPlotWindowRunning)
        self.actionExit.triggered.connect(self.exit)
    
        self.actionWriteTranscript.setShortcut(QtGui.QKeySequence("T"))
        self.actionShowPlot.setShortcut(QtGui.QKeySequence("R"))
        self.actionShowPlotRunning.setShortcut(QtGui.QKeySequence("L"))
        self.actionExit.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        
        self.shortcutPlay = QtWidgets.QShortcut(QtGui.QKeySequence("P"), self)
        self.shortcutPlay.activated.connect(self.playAudio)
        
        self.shortcutPrev = QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self)
        self.shortcutPrev.activated.connect(self.prevUtterance)
        
        self.shortcutNext = QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self)
        self.shortcutNext.activated.connect(self.nextUtterance)
        
    def showPlotWindow(self):
        self.plotWindow.show()
        self.plotWindow.replot(0)
    
    def showPlotWindowRunning(self):
        self.plotWindowRunning.show()
        self.plotWindowRunning.replot(0)
    
    def exit(self):
        self.statusBar().showMessage("Winding down...")
        del self.recordingManager
        sys.exit(0)
    
    def playAudioFile(self, audioFileName):
        self.stopAudio()
        subprocess.Popen(["aplay", audioFileName])
    
    def playParallelAudio(self):
        parallelAudio = self.recordingManager.getCurrentUtteranceParallelAudio()
        
        if parallelAudio != None:
            self.playAudioFile(parallelAudio)
            
    def stopAudio(self):
        subprocess.Popen(["killall", "aplay"])
    
    def startRecording(self):
        self.playParallelAudio()
        
        self.recordingManager.beginUtterance()
        self.statusBar().showMessage("Recording...")
        self.labelStatus.setText(glyphs["recording"])
        self.buttonRecord.setText("Recording... (release to stop)")
        
    def stopRecording(self):
        self.stopAudio()
        
        self.buttonRecord.setEnabled(False)
        self.statusBar().showMessage("Waiting for data...")
        hasNextUtt, emg, audio, self.lastUttEMGFile, self.lastUttAudioFile = self.recordingManager.endUtterance()
        self.plotWindow.plot(emg, audio)
        
        self.buttonRecord.setEnabled(True)
        self.buttonRecord.setText("Record (hold)")
        if hasNextUtt:
            self.displayReadyStatus()
            self.playParallelAudio()
        else:
            self.statusBar().showMessage("Final utterance recorded, wrote transcript.")
        self.updateGUI()
    
    def nextUtterance(self):
        self.recordingManager.nextUtterance()
        
        promptPos, promptCount = self.recordingManager.getPromptPosCount()
        self.displayReadyStatus()
        
        self.updateGUI()
    
    def prevUtterance(self):
        self.recordingManager.prevUtterance()
        
        promptPos, promptCount = self.recordingManager.getPromptPosCount()
        self.displayReadyStatus()
        
        self.updateGUI()
    
    def playAudio(self):
        uttFiles = self.recordingManager.getCurrentUtteranceFiles()
        if uttFiles is None:
            uttAudioFile = self.lastUttAudioFile
        else:
            uttAudioFile = uttFiles[1]
            
        if not uttAudioFile is None:
            self.playAudioFile(uttAudioFile)
            self.statusBar().showMessage("Playing audio")
        else:
            self.statusBar().showMessage("No audio file")
            
    def writeTranscript(self):
        self.recordingManager.writeTranscript()
        self.statusBar().showMessage("Wrote transcript to file")
    
