import os
import json
import random
import sys
import time
import soundfile
import glob
import subprocess

import numpy as np

sys.path.append("../EMGToSpeech")
from Nodes import UtteranceCutter
from Nodes import QuattrocentoDataSource
from Nodes import AudioSource
from Nodes import Receiver
from Nodes import LambdaNode
from Nodes import TDNCalculator
from Nodes import F0Calculator
from Nodes import DNN2
from Nodes import LPCNetSynth
from Nodes import SimpleSynth
from Nodes import JackAudioSink
from Nodes import RunningNorm

import numpy.random as npr

class RecordingManager:
    def __init__(self, basePath, speaker, session, uttList, recordingSetup, dummyRecord = False, parallelPath = None):
        self.dummyRecord = dummyRecord
        
        self.outDir = os.path.join(basePath, "{}_{}".format(speaker, session))
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)
        
        self.stateFileName = os.path.join(self.outDir, "recordingLog.json")
        self.state = {}
        if os.path.exists(self.stateFileName):
            stateFile = open(self.stateFileName, 'r')
            self.state = json.load(stateFile)
        else:
            promptList = open(uttList, 'r').readlines()
            promptList = list(zip(range(1, len(promptList) + 1), map(lambda x: x.strip(), promptList)))            
            random.shuffle(promptList)
            
            self.state["uttSets"] = None
            setFileName = uttList[:-4] + ".sets"
            if os.path.exists(setFileName):
                self.state["uttSets"] = {}
                setLines = open(setFileName, 'r').readlines()
                for line in setLines:
                    line = line.strip()
                    if len(line) != 0:
                        setInfo = line.split(" ")
                        self.state["uttSets"][setInfo[0]] = list(map(int, setInfo[1:]))
                    
            self.state["speaker"] = speaker
            self.state["session"] = session
            self.state["basePath"] = basePath
            
            self.state["promptList"] = promptList
            self.state["currUtterance"] = 0
            self.state["utteranceFiles"] = {}
            self.state["utteranceTimestamps"] = {}
            
            self.state["parallelPath"] = parallelPath
            
            with open(recordingSetup, 'r') as recordingSetupFile:
                recordingSetupStr = recordingSetupFile.read()
            self.state["recordingSetupStr"] = recordingSetupStr
        
        recordingSetupStr = self.state["recordingSetupStr"]
        
        if self.dummyRecord == False:
            self.emgDataSource = QuattrocentoDataSource.QuattrocentoDataSource.create_from_json(recordingSetupStr)
            self.audioDataSource = AudioSource.AudioSource()
            
            self.cutter = UtteranceCutter.UtteranceCutter(padding_ms = 400)([
                (self.emgDataSource, self.emgDataSource.acquisition_config["sampling_frequency"]),
                (self.audioDataSource, 16000),
            ])
            
            # Raw data receivers
            self.emg_receiver = Receiver.Receiver()(self.emgDataSource)
            self.audio_receiver = Receiver.Receiver()(self.audioDataSource)
            
            self.emgDataSource.start_processing()
            self.audioDataSource.start_processing()
            
        else:
            self.emgDataSource = QuattrocentoDataSource.QuattrocentoDataSource.create_from_json(recordingSetupStr, connect_immediately = False)
    
    def get_newest(self):
        """
        Return new data since last call
        """
        emg_data = np.concatenate(self.emg_receiver.get_data(True), axis = 0)
        audio_data = np.concatenate(self.audio_receiver.get_data(True), axis = 0)
        return (emg_data, audio_data)
    
    def __del__(self):
        if self.dummyRecord == True:
            return
        
        self.emgDataSource.stop_processing()
        self.audioDataSource.stop_processing()
        
    def writeState(self):
        stateFile = open(self.stateFileName, 'w')
        json.dump(self.state, stateFile)
    
    def beginUtterance(self):
        self.start_ts = time.time()
        if self.dummyRecord == True:
            return
        
        self.emgDataSource.set_trigger(True)
        self.cutter.begin_utterance()
    
    def endUtterance(self):
        if self.dummyRecord == False:
            self.emgDataSource.set_trigger(False)
            utteranceData = self.cutter.cut_utterance()
            self.cutter.clear_cutter()
        else:
            emgSf = self.emgDataSource.acquisition_config["sampling_frequency"]
            
            # Generate random utterance data
            fakeUttLen = npr.ranf() * 3.0 + 3.0
            utteranceData = [
                np.zeros((int(fakeUttLen * emgSf), len(self.emgDataSource.channel_set))),
                np.zeros((int(fakeUttLen * 16000.0), 2))
            ]
            
            sinePulse = np.sin(np.arange(0, np.pi, (np.pi / (16000.0 / 4.0))))
            channelSel =  npr.randint(0, utteranceData[1][:,0].shape[0], int(fakeUttLen * 4.0))
            utteranceData[1][channelSel, 0] = np.ones(len(channelSel))
            utteranceData[1][:, 0] = np.convolve(utteranceData[1][:, 0], sinePulse, mode='same')
            
            someNoise = npr.rand(int(fakeUttLen * 16000.0 / 50.0)) - 0.5
            fullNoise = np.interp(
                np.array(range(int(fakeUttLen * 16000.0))) / (fakeUttLen * 16000.0), 
                np.array(range(len(someNoise))) / float(len(someNoise)), 
                someNoise
            )
            utteranceData[1][:, 0] *= fullNoise * 25000.0
            utteranceData[1][:, 0] += fullNoise * 2500.0
            
            for dataRow in range(len(self.emgDataSource.channel_set)):
                sinePulse = np.sin(np.arange(0, np.pi, (np.pi / (emgSf / 64.0))))
                channelSel =  npr.randint(0, utteranceData[0][:, dataRow].shape[0], int(fakeUttLen * 4.0))
                utteranceData[0][channelSel, dataRow] = np.ones(len(channelSel))
                utteranceData[0][:, dataRow] = np.convolve(utteranceData[0][:, dataRow], sinePulse, mode='same')
                
                someNoise = npr.rand(int(fakeUttLen * emgSf / 50.0)) - 0.5
                fullNoise = np.interp(
                    np.array(range(int(fakeUttLen * emgSf))) / (fakeUttLen * emgSf), 
                    np.array(range(len(someNoise))) / float(len(someNoise)), 
                    someNoise
                )
                utteranceData[0][:, dataRow] *= fullNoise * 10.0
                utteranceData[0][:, dataRow] += fullNoise * 3.0

            
            # Generate fake markers
            fakeUttMarkerStart = fakeUttLen * 0.1
            fakeUttMarkerEnd = fakeUttLen * 0.9
            
            utteranceData[0][:,-1] = np.zeros(utteranceData[0][:,-1].shape)
            utteranceData[1][:,-1] = np.zeros(utteranceData[1][:,-1].shape)
            utteranceData[0][int(fakeUttMarkerStart * emgSf):int(fakeUttMarkerEnd * emgSf),-1] = np.ones(utteranceData[0][int(fakeUttMarkerStart * emgSf):int(fakeUttMarkerEnd * emgSf),-1].shape) * 10000.0
            utteranceData[1][int(fakeUttMarkerStart * 16000.0):int(fakeUttMarkerEnd * 16000.0),-1] = np.ones(utteranceData[1][int(fakeUttMarkerStart * 16000.0):int(fakeUttMarkerEnd * 16000.0),-1].shape) * 25000.0
            
        uttId = self.state["promptList"][self.state["currUtterance"]][0]
        uttIdStr = "{}_{}_{:04d}".format(
            self.state["speaker"], 
            self.state["session"],
            uttId
        )
        
        uttDir = os.path.join(self.outDir, uttIdStr)
        if not os.path.exists(uttDir):
            os.makedirs(uttDir)
        
        emgChannelCountStr = str(len(self.emgDataSource.channel_set))
        uttFiles = (
            os.path.join(uttDir, "emg_" + emgChannelCountStr + "ch_" + uttIdStr + ".adc"),
            os.path.join(uttDir, "audio_" + uttIdStr + ".wav"),
        )
        
        np.save(uttFiles[0] + ".npy", utteranceData[0])
        np.save(uttFiles[1] + ".npy", utteranceData[1])
        
        soundfile.write(uttFiles[0], utteranceData[0], self.emgDataSource.acquisition_config["sampling_frequency"], format = "RAW", subtype = 'PCM_16')
        soundfile.write(uttFiles[1], utteranceData[1] / float(2**15), 16000, format = "WAV", subtype = 'PCM_16')
        
        self.state["utteranceFiles"][str(uttId)] = uttFiles
        self.state["utteranceTimestamps"][str(uttId)] = self.start_ts
            
        hasNextFile = self.nextUtterance()
        if not hasNextFile:
            self.writeTranscript()
        self.writeState()
        return hasNextFile, utteranceData[0], utteranceData[1], uttFiles[0], uttFiles[1]
            
    def nextUtterance(self):
        self.state["currUtterance"] += 1
        if self.state["currUtterance"] >= len(self.state["promptList"]):
            self.state["currUtterance"] -= 1
            return False
        return True
        self.writeState()
            
    def prevUtterance(self):
        self.state["currUtterance"] -= 1
        if self.state["currUtterance"] < 0:
            self.state["currUtterance"] = 0
            return False
        return True
        self.writeState()
    
    def getPromptPosCount(self):
        return (self.state["currUtterance"], len(self.state["promptList"]))
    
    def getCurrentPrompt(self):
        return self.state["promptList"][self.state["currUtterance"]][1]
    
    def getCurrentUtteranceFiles(self):
        uttId = self.state["promptList"][self.state["currUtterance"]][0]
        if str(uttId) in self.state["utteranceFiles"]:
            return self.state["utteranceFiles"][str(uttId)]
        else:
            return None
    
    def getCurrentUtteranceParallelAudio(self):
        if not "parallelPath" in self.state or self.state["parallelPath"] == None:
            return None
        
        uttId = self.state["promptList"][self.state["currUtterance"]][0]
        uttIdStr = "*_*_{:04d}".format(uttId)
        
        uttDir = os.path.join(self.state["parallelPath"], uttIdStr)
        emgChannelCountStr = str(len(self.emgDataSource.channel_set))
        uttAudioPattern = os.path.join(uttDir, "audio_" + uttIdStr + ".wav")
        uttAudioFiles = glob.glob(uttAudioPattern)
        print(uttAudioPattern)
        if len(uttAudioFiles) != 1:
            return None
        uttAudioFile = uttAudioFiles[0]
        
        if os.path.exists(uttAudioFile):
            return uttAudioFile
        else:
            return None
    
    def writeTranscript(self):
        transcriptTemplate = "{{ID {id:}}} {{SPK_ID {spk:}}} {{UTT_NB {utt:04d}}} {{SESSION_NB {sess:}}} {{EMG_CH {{{channels:}}}}} {{SR_EMG {emg_sr_khz:}}} {{AUD_FILE {aud_file_name:}}} {{SR_AUDIO {aud_sr_khz:}}} {{EMG_FILE {emg_file_name:}}} {{AUDIBLE {audible_yesno:}}} {{TEXT {text:}}} {{UTT_TIMESTAMP {utt_ts:}}}"
        
        channelNames = []
        channelCounter = 0
        for channel in self.emgDataSource.channel_set:
            if channel in self.emgDataSource.get_input_channel_list(self.emgDataSource.acquisition_config["input_set"], "trigger"):
                channelNames.append("MARKER")
                continue
            
            if channel in self.emgDataSource.get_input_channel_list(self.emgDataSource.acquisition_config["input_set"], "sampcount"):
                channelNames.append("SAMP_COUNT")
                continue
                
            if channel in self.emgDataSource.get_input_channel_list(self.emgDataSource.acquisition_config["input_set"], "buffer"):
                channelNames.append("BUFFER")
                continue
            
            channelNames.append("EMG_" + str(channelCounter))
            channelCounter += 1
        
        emg_sr_hz = self.emgDataSource.acquisition_config["sampling_frequency"]
        aud_sr_hz = 16000
        
        transcript = ""
        for uttId, uttText in self.state["promptList"]:
            if not str(uttId) in self.state["utteranceFiles"]:
                continue
            
            uttFiles = self.state["utteranceFiles"][str(uttId)]
            uttIdStr = "{}_{}_{:04d}".format(
                self.state["speaker"], 
                self.state["session"],
                uttId
            )
            transcriptLine = transcriptTemplate.format(
                id = uttIdStr,
                spk = self.state["speaker"],
                sess = self.state["session"],
                utt = uttId,
                channels = " ".join(channelNames),
                emg_sr_khz = str(emg_sr_hz / 1000.0),
                aud_sr_khz = str(aud_sr_hz / 1000.0),
                emg_file_name = os.path.basename(uttFiles[0]),
                aud_file_name = os.path.basename(uttFiles[1]),
                audible_yesno = "YES",
                text = uttText,
                utt_ts = self.state["utteranceTimestamps"][str(uttId)]
            )
            transcript += transcriptLine + "\n"
        
        transcriptFileName = os.path.join(self.outDir, "transcript_{}_{}".format(self.state["speaker"], self.state["session"]))
        with open(transcriptFileName, 'w') as transcriptFile:
            transcriptFile.write(transcript)
        
        if not self.state["uttSets"] is None:
            for uttSet in self.state["uttSets"].keys():
                uttSetFile = open(os.path.join(self.outDir, uttSet + "_set"), 'w')
                for uttId, uttText in self.state["promptList"]:
                    if not str(uttId) in self.state["utteranceFiles"]:
                        continue
                    
                    if not int(uttId) in self.state["uttSets"][uttSet]:
                        continue
                    
                    uttIdStr = "{}_{}_{:04d}".format(
                        self.state["speaker"], 
                        self.state["session"],
                        uttId
                    )
                    uttSetFile.write(uttIdStr + "\n")
                uttSetFile.close()
                
