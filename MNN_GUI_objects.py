'''
Author: Daniel Del Hoyo Gomez
Universidad Politécnica de Madrid
Simple GUI designed to work with the Metabolic Neural Network (MNN) objects.
'''

import tkinter as tk
from tkinter import ttk
from MNN_objects import *
import os
import time

class DatasetFrame(ttk.Frame):
    '''Tab for the dataset creation and preprocessing'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Main options
        self.optionsFrame = tk.LabelFrame(self, text='Main options', pady=10)
        self.optionsFrame.pack(fill=tk.X)
        self.statesFrame = ttk.Frame(self.optionsFrame)
        self.statesFrame.pack(fill=tk.X)
        self.newState = tk.BooleanVar(self.statesFrame)
        self.newState.set(True) #set check state
        self.newCheck = tk.Checkbutton(self.statesFrame, text='New Dataset', var=self.newState)
        self.newCheck.pack(side='left')
        self.prepState = tk.BooleanVar(self.statesFrame)
        self.prepState.set(True) #set check state
        self.prepButton = tk.Checkbutton(self.statesFrame, text='Do Preprocess', var=self.prepState)
        self.prepButton.pack(side='left', padx=10)
        self.nameFrame = ttk.Frame(self.optionsFrame)
        self.nameFrame.pack(fill=tk.X)
        self.nameLabel = ttk.Label(self.nameFrame, text = '- Dataset name: ')
        self.nameLabel.pack(side = 'left')
        self.nameEntry = ttk.Entry(self.nameFrame)
        self.nameEntry.pack(side = 'left', padx=10)
        self.nameEntry.insert(0,'simpleEC')

        #New dataset options
        self.newFrame = tk.LabelFrame(self, text='New dataset', pady=10)
        self.newFrame.pack(fill=tk.X)
        self.states2Frame = ttk.Frame(self.newFrame)
        self.states2Frame.pack(fill=tk.X)
        self.doFBAState = tk.BooleanVar(self.states2Frame)
        self.doFBAState.set(False) #set check state
        self.doFBACheck = tk.Checkbutton(self.states2Frame, text='Do FBA', var=self.doFBAState)
        self.doFBACheck.pack(side='left')
        self.numFrame = ttk.Frame(self.newFrame)
        self.numFrame.pack(fill=tk.X)
        self.numLabel = ttk.Label(self.numFrame, text = '- Number of FBAs: ')
        self.numLabel.pack(side = 'left')
        self.numEntry = ttk.Entry(self.numFrame)
        self.numEntry.pack(side = 'left', padx=10)
        self.numEntry.insert(0,'5000')
        self.folderFrame = ttk.Frame(self.newFrame)
        self.folderFrame.pack(fill=tk.X)
        self.folderLabel = ttk.Label(self.folderFrame, text = '- Data folder: ')
        self.folderLabel.pack(side = 'left')
        self.folderEntry = ttk.Entry(self.folderFrame)
        self.folderEntry.pack(side = 'left', padx=10, pady=5)
        self.folderEntry.insert(0,'biggModels/')
        self.modelsFrame = ttk.Frame(self.newFrame)
        self.modelsFrame.pack(fill=tk.X)
        self.modelsLabel = ttk.Label(self.modelsFrame, text = '- Model names (csv): ')
        self.modelsLabel.pack(side = 'left')
        self.modelsEntry = ttk.Entry(self.modelsFrame)
        self.modelsEntry.pack(side = 'left', padx=10, pady=5)
        self.modelsEntry.insert(0,'iWFL,iJO1366')

        #Preprocessing options
        self.prepFrame = tk.LabelFrame(self, text='Preprocessing', pady=10)
        self.prepFrame.pack(fill=tk.X)
        self.thrFrame = ttk.Frame(self.prepFrame)
        self.thrFrame.pack(fill=tk.X)
        self.thrLabel = ttk.Label(self.thrFrame, text = '- Filtering threshold: ')
        self.thrLabel.pack(side = 'left')
        self.thrEntry = ttk.Entry(self.thrFrame)
        self.thrEntry.pack(side = 'left', padx=10)
        self.thrEntry.insert(0,'0.05')
        
        self.makeButton = ttk.Button(self, text=" Make dataset ", command=self.makeDataset)
        self.makeButton.pack(side='bottom')
        
        self.greet_label = ttk.Label(self)
        self.greet_label.pack(pady=10)

    def parseModels(self):
        '''Parse the models entered in the entry
        '''
        strModels = self.modelsEntry.get()
        return strModels.split(',')

    def makeDataset(self):
        '''Action to make the dataset after parsing all the inputs'''
        self.newDataset, self.doPreprocess = self.newState.get(), self.prepState.get()
        self.datasetName = self.nameEntry.get()
        self.doFBA, self.nFBAs = self.doFBAState.get(), int(self.numEntry.get())
        self.modelsFolder = self.folderEntry.get()
        self.modelBases = self.parseModels()
        #Create / Load dataset
        self.dataset = MNN_dataset()
        self.dataset.loadDataset(self.datasetName, self.nFBAs, self.modelBases,self.newDataset,
                                 self.doFBA, self.modelsFolder)
        #Preprocess dataset
        if self.doPreprocess:
            self.threshold = float(self.thrEntry.get())
            self.dataset.preprocessDataset(self.threshold)
        
        lbl='''Dataset shapes (Original / Preprocessed):\n\nNumber of FBAs:\t\t{}\nInput shapes:\t\t{}  /  {}\nReactions shapes:\t\t{}  /  {}\nExchanges shapes:\t{}  /  {}\
             '''.format(self.dataset.netInps.shape[0], self.dataset.oriShapes[0], self.dataset.prepShapes[0],
                     self.dataset.oriShapes[1], self.dataset.prepShapes[1], self.dataset.oriShapes[2], self.dataset.prepShapes[2])
        self.greet_label["text"] = lbl

        
class NetworkFrame(ttk.Frame):
    '''Tab for the MNN system creation, training, testing and predicting'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #Load dataset
        self.datasetFrame = tk.LabelFrame(self, text='Load Dataset', pady=5)
        self.datasetFrame.pack(fill=tk.X)
        self.nameFrame = ttk.Frame(self.datasetFrame)
        self.nameFrame.pack(fill=tk.X)
        self.nameLabel = ttk.Label(self.nameFrame, text = '- Dataset name: ')
        self.nameLabel.pack(side = 'left')
        self.nameEntry = ttk.Entry(self.nameFrame)
        self.nameEntry.pack(side = 'left', padx=10)
        self.nameEntry.insert(0,'prep_simpleEC')
        self.dataFrame = ttk.Frame(self.datasetFrame)
        self.dataFrame.pack(fill=tk.X)
        self.loadButton = tk.Button(self.dataFrame, text=' Load ', command = self.loadDataset)
        self.loadButton.pack(side = 'left', padx=10)
        self.loadLabel = tk.Label(self.dataFrame, text = "")
        self.loadLabel.pack(side = 'left', padx=10)
         
        #Create / Load MNN
        self.loadFrame = tk.LabelFrame(self, text='Create / Load MNN', pady=5)
        self.loadFrame.pack(fill=tk.X)
        self.nameFrame = ttk.Frame(self.loadFrame)
        self.nameFrame.pack(fill=tk.X)
        self.versionFrame = ttk.Frame(self.nameFrame)
        self.versionFrame.pack(fill=tk.X)
        self.sparseState = tk.BooleanVar(self.nameFrame)
        self.sparseState.set(True) #set check state
        self.sparseButton = tk.Checkbutton(self.nameFrame, text='Sparse', var=self.sparseState)
        self.sparseButton.pack(side='left', padx=10)
        self.lblP = tk.Label(self.nameFrame, text="Version: ")
        self.lblP.pack(side='left')
        self.versionBox = ttk.Combobox(self.nameFrame)
        self.versionBox.pack(side='left')
        self.versionBox['values'] = ['11 - Conventional', '21 - GRRANN', '22 - HBB',
                                     '23 - HBB_Shutdown', '24 - Reiterative']
        self.versionBox.current(0)
        self.createFrame = ttk.Frame(self.loadFrame)
        self.createFrame.pack(fill=tk.X)
        self.createButton = tk.Button(self.createFrame, text=' Create ', command = self.createMNN, state = 'disabled')
        self.createButton.pack(side = 'left', padx=10)
        self.browButton = tk.Button(self.createFrame, text = "Browse A File", command = self.findMNN, state = 'disabled')
        self.browButton.pack(side='left')
        self.fileLabel = tk.Label(self.createFrame, text = "")
        self.fileLabel.pack(side='left', padx=10)

        #Divide dataset
        self.divideFrame = tk.LabelFrame(self, text='Divide dataset', pady=5)
        self.divideFrame.pack(fill=tk.X)
        self.propFrame = ttk.Frame(self.divideFrame)
        self.propFrame.pack(fill=tk.X)
        self.nLabel = ttk.Label(self.propFrame, text = '- Proportions ')
        self.nLabel.pack(side = 'left')
        self.trainLabel = ttk.Label(self.propFrame, text = 'Train: ')
        self.trainLabel.pack(side = 'left')
        self.trainEntry = ttk.Entry(self.propFrame, width=8)
        self.trainEntry.pack(side = 'left', padx=5)
        self.trainEntry.insert(0,'0.8')
        self.valLabel = ttk.Label(self.propFrame, text = ' Validation: ')
        self.valLabel.pack(side = 'left')
        self.valEntry = ttk.Entry(self.propFrame, width=8)
        self.valEntry.pack(side = 'left', padx=5)
        self.valEntry.insert(0,'0.1')
        self.testLabel = ttk.Label(self.propFrame, text = ' Test: ')
        self.testLabel.pack(side = 'left')
        self.testEntry = ttk.Entry(self.propFrame, width=8)
        self.testEntry.pack(side = 'left', padx=5)
        self.testEntry.insert(0,'0.1')
        self.divFrame = ttk.Frame(self.divideFrame)
        self.divFrame.pack(fill=tk.X)
        self.divButton = tk.Button(self.divFrame, text=' Divide ', command = self.divideData, state='disabled')
        self.divButton.pack(side = 'left', padx=10, pady=5)
        self.divLabel = tk.Label(self.divFrame, text = "")
        self.divLabel.pack(side = 'left', padx=10)

        #Train MNN
        self.trainFrame = tk.LabelFrame(self, text='Train MNN', pady=5)
        self.trainFrame.pack(fill=tk.X)
        self.epFrame = ttk.Frame(self.trainFrame)
        self.epFrame.pack(fill=tk.X)
        self.epLabel = ttk.Label(self.epFrame, text = '- Epochs: ')
        self.epLabel.pack(side = 'left')
        self.epEntry = ttk.Entry(self.epFrame, width=8)
        self.epEntry.pack(side = 'left', padx=5)
        self.epEntry.insert(0,'10')
        self.btFrame = ttk.Frame(self.trainFrame)
        self.btFrame.pack(fill=tk.X)
        self.trainButton = tk.Button(self.btFrame, text=' Train ', command = self.trainMNN, state='disabled')
        self.trainButton.pack(side = 'left', padx=10, pady=5)
        self.saveButton = tk.Button(self.btFrame, text=' Save ', command = self.saveMNN, state='disabled')
        self.saveButton.pack(side = 'left', padx=10, pady=5)
        self.trainLabel = tk.Label(self.btFrame, text = "")
        self.trainLabel.pack(side = 'left', padx=10)
        #Plot training
        self.plotFrame = ttk.Frame(self.trainFrame)
        self.plotFrame.pack(fill=tk.X)
        self.plossButton = tk.Button(self.plotFrame, text=' Plot Loss ', command = self.plotLoss, state='disabled')
        self.plossButton.pack(side = 'left', padx=10)
        self.pMAEButton = tk.Button(self.plotFrame, text=' Plot MAE ', command = self.plotMAE, state='disabled')
        self.pMAEButton.pack(side = 'left', padx=10)

        #Test MNN
        self.testFrame = tk.LabelFrame(self, text='Test MNN', pady=5)
        self.testFrame.pack(fill=tk.X)
        self.teFrame = ttk.Frame(self.testFrame)
        self.teFrame.pack(fill=tk.X)
        self.teLabel = tk.Label(self.teFrame, text = "- Test on divided dataset")
        self.teLabel.pack(side = 'left')
        self.teButton = tk.Button(self.teFrame, text=' Test on divided ', command = self.testDivided, state='disabled')
        self.teButton.pack(side = 'left', padx=10, pady=5)
        self.teresLabel = tk.Label(self.teFrame, text = "")
        self.teresLabel.pack(side = 'left', padx=10)
        self.te2Frame = ttk.Frame(self.testFrame)
        self.te2Frame.pack(fill=tk.X)
        self.te2Label = tk.Label(self.te2Frame, text = "- Test extra dataset")
        self.te2Label.pack(side = 'left')
        self.te2Entry = ttk.Entry(self.te2Frame)
        self.te2Entry.pack(side = 'left', padx=10)
        self.te2Entry.insert(0,'iEC042_toy')
        self.te2Button = tk.Button(self.te2Frame, text=' Test extra ', command = self.testExtra, state='disabled')
        self.te2Button.pack(side = 'left', padx=10)
        self.te2resLabel = tk.Label(self.te2Frame, text = "")
        self.te2resLabel.pack(side = 'left', padx=10)

        #Predict with MNN
        self.predFrame = tk.LabelFrame(self, text='Predict with MNN', pady=5)
        self.predFrame.pack(fill=tk.X)
        self.prFrame = ttk.Frame(self.predFrame)
        self.prFrame.pack(fill=tk.X)
        self.priLabel = tk.Label(self.prFrame, text = "- Load extra inputs")
        self.priLabel.pack(side = 'left')
        self.brow2Button = tk.Button(self.prFrame, text = "Browse A File", command = self.findInputs, state = 'disabled')
        self.brow2Button.pack(side='left', padx=10)
        self.prButton = tk.Button(self.prFrame, text=' Predict ', command = self.predict, state='disabled')
        self.prButton.pack(side = 'left', padx=10)
        self.prLabel = tk.Label(self.predFrame, text = "")
        self.prLabel.pack(side = 'left', padx=10)

        self.loadOFrame = ttk.Frame(self.predFrame)
        self.loadOFrame.pack(fill=tk.X)
        self.loadOLabel = tk.Label(self.loadOFrame, text="- Load extra outputs")
        self.loadOLabel.pack(side='left')
        self.loadOButton = tk.Button(self.loadOFrame, text="Browse A File", command=self.findOutputs)#, state='disabled')
        self.loadOButton.pack(side='left', padx=10)
        self.heatButton = tk.Button(self.loadOFrame, text=' Heatmap ', command=self.heatmap, state='disabled')
        self.heatButton.pack(side='left', padx=10)
        self.heatLabel = tk.Label(self.predFrame, text="Sample index")
        self.heatLabel.pack(side='left', padx=10)
        self.heatEntry = ttk.Entry(self.predFrame)
        self.heatEntry.pack(side='left', padx=10)
        self.heatEntry.insert(0, '1')

    ######################################
    ######### Button functions ###########
    ######################################

    def loadDataset(self):
        '''Load the dataset to specify the network structure
        '''
        self.datasetName = self.nameEntry.get()
        self.data = MNN_dataset()
        self.data.loadDataset(self.datasetName)
        
        self.loadLabel['text'] = 'Dataset {} loaded'.format(self.datasetName)
        self.createButton['state'], self.browButton['state'] = 'active', 'active'

    def createMNN(self):
        '''Creates a MNN with the given parameters and the structure of the loaded dataset
        '''
        #Build the network model
        sparse = self.sparseState.get()
        version = int(self.versionBox.get().split(' -')[0])
        nreit = 1 if version==24 else 0

        self.net = MNN_net(self.data)
        self.net.createMNN(version, sparse, save=True)

        if self.net.version in [11, 21]:
            self.fileLabel.configure(text = 'MNN dataset:  {}\nVersion:  {}\nSparse:  {}'.format(self.data.datasetName, self.net.version, False))
        else:
            self.fileLabel.configure(text = 'MNN dataset:  {}\nVersion:  {}\nSparse:  {}'.format(self.data.datasetName, self.net.version, self.net.sparse))
        self.divButton['state'], self.te2Button['state'], self.brow2Button['state'] = 'active', 'active', 'active'
        
    def findMNN(self):
        '''File dialog to look for a pretrained MNN weights file (h5)
        '''
        self.net = MNN_net(self.data)
        self.loadMNN()
        self.fileLabel.configure(text = 'MNN file:  {}\nVersion:  {}\nSparse:  {}'\
                                 .format(self.net.mainDataFile.split('/')[-1], self.net.version, self.net.sparse))
        time.sleep(0.1)

    def divideData(self):
        '''Divide the dataset which has been loaded'''
        #Get parameters
        trainProp, valProp = float(self.trainEntry.get()), float(self.valEntry.get())
        self.net.divideData(trainProp, valProp)

        self.trainButton['state'], self.teButton['state'] = 'active', 'active'
        print('Dataset divided')
        self.divLabel['text'] = 'Dataset divided'

    def trainMNN(self):
        '''Trains the MNN which has been created or loaded with the loaded dataset
        '''
        epochs=int(self.epEntry.get())
        self.net.trainMNN(epochs)
        self.reactTrError, self.exchTrError, self.reactValError, self.exchValError = self.net.getErrors()

        if self.net.version == 11:
            self.trainLabel['text'] = 'Exchanges error: {:f}'.format(self.exchValError)
        else:
            self.trainLabel['text'] = 'Reactions error: {:f}\nExchanges error: {:f}'\
                                      .format(self.reactValError, self.exchValError)
        self.saveButton['state'], self.plossButton['state'], self.pMAEButton['state'] = 'active', 'active', 'active'

    def saveMNN(self):
        '''Saves the MNN'''
        filename = self.net.saveNet()

    def plotLoss(self):
        '''Plot a graph with the loss in the training process'''
        self.net.plotLoss()

    def plotMAE(self):
        '''Plot a graph with the MAE in the training process'''
        self.net.plotMAE()

    def testDivided(self):
        '''Evaluates the test set created in the division of the original dataset'''
        self.net.testDivided()
        self.printRes(self.net.testDivRes, label=self.teresLabel)
           
    def testExtra(self):
        '''Evaluates the extra dataset which has been loaded
        '''
        self.net.loadExtraDataset(self.te2Entry.get())
        #Adjust the outputs
        self.net.testExtraDataset()

        self.printRes(self.net.testExtraRes, label=self.te2resLabel)

    def findInputs(self):
        '''File dialog to look for an inputs.pickle file. It needs to have an associated parameters.pickle file
        '''
        self.net.loadInputs()
        self.prButton['state'] = 'active'
        time.sleep(0.1)

    def findOutputs(self):
        '''File dialog to look for an outputs.pickle file. It needs to have an associated parameters.pickle file
        '''
        self.net.loadOutputs()
        self.heatButton['state'] = 'active'
        time.sleep(0.1)
    
    def predict(self):
        '''Predicts the outputs of loaded inputs
        '''
        self.net.predict()

    def heatmap(self):
        '''Produces a heatmap to compare the outputs loaded with the predicted
        '''
        self.net.heatmap(int(self.heatEntry.get()))

    ###############################################
    ########### Auxiliar functions ################
    ###############################################

    def printRes(self, res, label):
        '''Modify a label with the results
        '''
        if self.net.version==11:
            reactError, exchError = None, res[-1]
            label['text'] = 'Exchanges error: {:f}'.format(exchError)
        else:
            reactError, exchError = res[-2], res[-1]
            label['text'] = 'Reactions error: {:f}\nExchanges error: {:f}'\
                            .format(reactError, exchError) 

    def loadMNN(self):
        '''Load the MNN weights stored in a h5 file
        '''
        self.net.loadMNN()
        self.divButton['state'] = 'active'
        self.te2Button['state'] = 'active'
        self.brow2Button['state'] = 'active'
        
    
class Application(ttk.Frame):
    def __init__(self, main_window):
        super().__init__(main_window)
        main_window.title("Metabolic Neural Networks")
        
        self.notebook = ttk.Notebook(self)
        self.dataset_frame = DatasetFrame(self.notebook)
        self.notebook.add(self.dataset_frame, text="Dataset", padding=10)
        self.network_frame = NetworkFrame(self.notebook)
        self.notebook.add(self.network_frame, text="Network", padding=10)
        self.notebook.pack(padx=10, pady=10)
        self.pack()

def __main__():
    main_window = tk.Tk()
    app = Application(main_window)
    app.mainloop()

if __name__ == "__main__":
    __main__()
