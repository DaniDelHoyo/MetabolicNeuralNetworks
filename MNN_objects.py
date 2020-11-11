'''
Author: Daniel Del Hoyo Gomez
Universidad Politécnica de Madrid
Metabolic Neural Network (MNN) objects.
'''

from modulesDatasets import *
from modulesNetworks import *
from tkinter import filedialog

class MNN_dataset(object):
    def __init__(self):
        self.loaded = False
        self.oriShapes, self.prepShapes = 0, 0
        self.totalFBAs = 0

    def getOriginalShape(self):
        '''Return the shapes of the database before the preprocessing'''
        return self.oriShapes

    def getPreprocessedShape(self):
        '''Return the shapes of the database after the preprocessing'''
        return self.prepShapes

    def getDatasetName(self):
        '''Return the name of the dataset'''
        return self.datasetName

    def help(self):
        print('''MNN_dataset object with two principal functions:)
        1) loadDataset: Load or creates a FBA dataset readable by MNN object)
        \tInputs: datasetName: str, name of the dataset to load or to be created
        \t\tnFBAs: int, number of FBAs per Genome Scale Model (GEM) to use in the dataset
        \t\tmodelBases: list of str, names of the GEMs to introduce in the system
        \t\tnewDataset: boolean, whether to create a new dataset or load
        \t\tdoFBAs: boolean, whether to create new FBA instances to introduce in the dataset
        \n\t2) preprocessDataset: preprocessed the loaded dataset by normalizing and filtering it
        \tInputs: threshold: float, minimum proportion of non-null values to consider a reaction flux
        ''')

    def loadDataset(self, datasetName, nFBAs = 5000, modelBases = ['iJO1366', 'iYS1720'],
                    newDataset = False, doFBA = False):
        '''Create or load the MNN dataset. For loading, just the datasetName is neccesary.
        - datasetName: str, name of the dataset. Ex: ecSalm for ecSalm_inputs.pickle file
        - nFBAs: int, number of FBAs per model to include in the dataset
        - modelBases: list of str, names of the GEM BIGG models to include
        - newDataset: boolean, whether to create a new dataset or load
        - doFBAs: boolean, whether to create new FBA instances to introduce in the dataset
        '''
        self.datasetName, self.nFBAs = datasetName, nFBAs
        if newDataset or not datasetName+'_inputs.pickle' in os.listdir():
            #Load and/or run the FBAs of the selected cobra models
            self.netInps, self.netOuts, self.pDic = createFBADataset(modelBases, nFBAs, doFBA)
            nFBAs = self.netInps.shape[0]
            self.pDic.update({'sInpStrains': getSparseInputStrains(self.netOuts[0], nFBAs),
                 'nFBAs': nFBAs, 'modelBases': modelBases})
            #Save produced dataset and parameters
            saveInOut(self.netInps, self.netOuts, self.datasetName, append = False)
            saveParametersDic(self.pDic, self.datasetName)
        else:
            #Load preexistent dataset and parameters
            self.netInps, self.netOuts = loadInOut(self.datasetName)
            self.pDic = loadParametersDic(self.datasetName)
            self.nFBAs = self.pDic['nFBAs']
        self.oriShapes = [self.netInps.shape[1], self.netOuts[0].shape[1], self.netOuts[-1].shape[1]]
        self.prepShapes = [self.netInps.shape[1], self.netOuts[0].shape[1], self.netOuts[-1].shape[1]]
        self.loaded = True
        self.totalFBAs = self.netInps.shape[0]

    def preprocessDataset(self, threshold = 0.05):
        '''Preprocess dataset by filtering the null variables and normalizing it
        - threshold: float, minimum proportion of non-null values to consider a reaction flux
        '''
        if self.loaded:
            #Filtering the output
            self.netOuts, self.pDic = filterOutputs(self.netOuts, self.pDic, threshold)
            oriNetOuts = [self.netOuts[0].copy(), self.netOuts[1].copy()]
            #Normalizing inputs and outputs
            self.netInps /= 1000
            outMaxs, outMins = [[],[]], [[],[]]
            for i in range(2):
                self.netOuts[i], outMaxs[i], outMins[i] = norm_outputs_withZeros(self.netOuts[i])
            normZeros = real2norm(np.zeros((1,oriNetOuts[0].shape[1])), outMaxs[0], outMins[0])
            self.pDic.update({'outMaxs': outMaxs, 'outMins': outMins, 'normZeros': normZeros})

            #Save preprocessed dataset and parameters
            saveInOut(self.netInps, self.netOuts, 'prep_'+self.datasetName, append = False)
            saveParametersDic(self.pDic, 'prep_'+self.datasetName)
            self.prepShapes = [self.netInps.shape[1], self.netOuts[0].shape[1], self.netOuts[-1].shape[1]]

        else:
            print('A dataset has not been loaded yet')


class MNN_net(object):
    def __init__(self, mnnDataset):
        self.mainDataset = mnnDataset
        self.predBase, self.newDatasetName = '', ''

    def createMNN(self, version=22, sparse=False, save=True, saveFolder='savedSystems/'):
        '''Creates a new MNN system with the structure of the preloaded dataset
        - version: int, version of the MNN
        #MNN version: 11=Conventional, 21=GRRANN, 22=HBB, 23=HBB_Shutdown, 24=Reiterative_Shutdown
        - sparse: boolean, whether to use the sparse connections in the system
        - save: boolean, whether to save the created model
        - saveFolder: str, folder where the system must be saved
        '''
        data = self.mainDataset
        self.sparse, self.version = sparse, version
        self.nreit = 1 if self.version==24 else 0
        normZeros, sInpStrains = data.pDic['normZeros'], data.pDic['sInpStrains']
        mediumMetabs, glayersIds = data.pDic['mediumMetabs'], data.pDic['glayersIds']
        if self.sparse:
            self.matConnections = getConnectionsMatrix(data.pDic['gLayDics'])
        else:
            self.matConnections = []
        self.modelMNN = buildNetwork(self.version, mediumMetabs, glayersIds, normZeros, self.nreit, self.matConnections)
        if save:
            self.saveNet()

    def loadMNN(self, filename=None, key=None):
        '''Load a preexistent MNN system
        - filename: str, name of the h5 file with the MNN weights stored
        - key: str, key with the version+sparse of the system to load. Default: parsed from the filename
        '''
        if filename==None:
            filename = filedialog.askopenfilename(
            initialdir =  os.getcwd(), title = "Select A File",
            filetype = (("h5 files","*.h5"),("all files","*.*")))
        if key==None:
            key = filename.split('/')[-1].split('v')[-1].split('.')[0]
        self.mainDataFile = filename
        self.version, self.sparse = versionFromKey(key)
        self.nreit = 1 if self.version==24 else 0
        self.modelMNN = self.buildFromKey(key)
        self.modelMNN.load_weights(filename)

    def divideData(self, trainProp, valProp):
        '''Divide the dataset which has been loaded into training, validation and test sets.
        - trainProp: proportion for the training set
        - valProp: proportion for the validation set (test set proportion is the rest)
        '''
        data = self.mainDataset
        inpStrains = getTotalInputStrains(data.pDic['sInpStrains'], data.pDic['nFBAs'])

        #Shuffle the dataset
        inout = shuffle_dataset([data.netInps, inpStrains, *data.netOuts])
        netInps, inpStrains, netOuts = inout[0], inout[1], inout[2:]
        #Adjust the outputs
        netOuts, self.nreit = adjustOutput(self.version, netOuts, self.nreit)
        
        #Divide datasets
        x, y  = divideSetsVersion(self.version, netInps, netOuts, inpStrains, trainProp, valProp)
        [self.xTrain, self.xVal, self.xTest], [self.yTrain, self.yVal, self.yTest] = x, y

    def trainMNN(self, epochs=10, verb=2):
        '''Trains the MNN which has been created or loaded with the loaded dataset
        - epochs: int, number of epochs to train the system
        '''
        self.hist = self.modelMNN.fit(self.xTrain, self.yTrain, epochs = epochs, verbose=verb,
                                      validation_data=(self.xVal, self.yVal))
        self.reactTrError, self.exchTrError, self.reactValError, self.exchValError = self.getErrors()
        self.printTrainErrors()

    def plotLoss(self):
        '''Plot a graph with the loss in the training process'''
        plot_output(self.version, self.hist, self.nreit, which=0)

    def plotMAE(self):
        '''Plot a graph with the MAE in the training process'''
        plot_output(self.version, self.hist, self.nreit, which=1)

    def testDivided(self):
        '''Evaluates the test set created in the division of the main dataset'''
        self.testDivRes = evaluateModel(self.modelMNN, self.xTest, self.yTest)
        self.printTestedErrors(self.testDivRes)

    def loadExtraDataset(self, newDatasetName):
        '''Load an extra dataset in order to test the current system on it.
        The extra dataset is adapted to the structure of the system
        - newDatasetName: str, name of the new dataset
        '''
        data = self.mainDataset
        self.newDatasetName = newDatasetName
        self.new_pDic = loadParametersDic(newDatasetName)
        newInps, newOuts = loadInOut(newDatasetName)
        newInps, self.realNewOuts = adaptInputOutputMultiStrain(newInps, newOuts, self.new_pDic,
                                                                data.pDic)
        self.new_pDic['sInpStrains'] = getInputStrain(self.realNewOuts[0])
        self.newInps = newInps / 1000
        self.newOuts = [real2normArray(self.realNewOuts[0], data.pDic['outMaxs'][0], data.pDic['outMins'][0]),
                        real2normArray(self.realNewOuts[1], data.pDic['outMaxs'][1], data.pDic['outMins'][1])]

    def testExtraDataset(self):
        '''Test the MNN system in the loaded extra dataset'''
        if self.newDatasetName != '':
            #Adjust the outputs
            inpStrains = getTotalInputStrains(self.new_pDic['sInpStrains'], self.new_pDic['nFBAs'])
            newOuts, self.nreit = adjustOutput(self.version, self.newOuts, self.nreit)
            x, y  = divideSetsVersion(self.version, self.newInps, newOuts, inpStrains, 0, 0)
            [_, _, self.new_xTest], [_, _, self.new_yTest] = x, y
            #Test new dataset
            self.testExtraRes = evaluateModel(self.modelMNN, self.new_xTest, self.new_yTest)
            self.printTestedErrors(self.testExtraRes)
        else:
            print('\nAn extra dataset must be loaded first ( self.loadExtraDataset(newDatasetName) )')

    def loadInputs(self, pickleFilename=None):
        '''Load inputs for making predictions with the MNN system
        - pickleFilename: str, filename of the inputs file (in pickle format and with correspondent parameters file)
        '''
        if pickleFilename==None:
            self.predFilename = filedialog.askopenfilename(
                initialdir =  os.getcwd(), title = "Select A File",
                filetype = (("input pickle files","*.pickle"),("all files","*.*")))
        else:
            self.predFilename = pickleFilename
        infn, self.predBase = self.predFilename, self.predFilename.split('_inputs')[0]
        with open(infn,'rb') as finputs:
            self.predInps = pickle.load(finputs)
        self.pred_pDic = loadParametersDic(self.predBase)
        #Adapt and normalize
        self.predInps = adaptInputMultiStrain(self.predInps, self.pred_pDic, self.mainDataset.pDic)
        self.predInps /= 1000

    def predict(self):
        '''Generates the predictions for the loaded inputs'''
        if self.predBase != '':
            #Adjust to input MNN structure
            predInps = prepareInputs(self.predInps, self.version, self.mainDataset.pDic)
            self.predOuts = self.modelMNN.predict(predInps)
            #Denormalize predictions
            data = self.mainDataset
            outMaxs, outMins = data.pDic['outMaxs'], data.pDic['outMins']
            if self.version == 11:
                self.realPredOuts = [norm2real(self.predOuts[-1], outMaxs[-1], outMins[-1])]
            else:
                self.realPredOuts = [norm2real(self.predOuts[-2], outMaxs[-2], outMins[-2]),
                                     norm2real(self.predOuts[-1], outMaxs[-1], outMins[-1])]
            #Save predictions
            outfn = self.predBase + '_v' + str(self.version) + '_predictions.pickle'
            with open(outfn, 'wb') as handle:
                pickle.dump(self.realPredOuts, handle)
            print('\nPredictions saved in '+outfn)
        else:
            print('\nA prediction Input must be loaded first ( self.loadInputs(pickleFilename) )')

    ############################
    #### Auxiliar functions ####
    ############################

    def buildFromKey(self, key):
        '''Build a MNN using the key in the filename (version and sparse)
        - key: str, MNN version + sparsity (Ex: 22True)
        '''
        dataset = self.mainDataset
        self.version, self.sparse = versionFromKey(key)
        normZeros, sInpStrains = dataset.pDic['normZeros'], dataset.pDic['sInpStrains']
        mediumMetabs, glayersIds = dataset.pDic['mediumMetabs'], dataset.pDic['glayersIds']

        #Sparse or Full version
        self.nreit = 1 if self.version==24 else 0
        if self.sparse:
            matCon = getConnectionsMatrix(dataset.pDic['gLayDics'])
        else:
            matCon = []
        #MNN version: 11=Conventional, 21=GRRANN, 22=HBB, 23=HBB_Shutdown, 24=Reiterative_Shutdown
        self.modelMNN = buildNetwork(self.version, mediumMetabs, glayersIds, normZeros, self.nreit, matCon)
        return self.modelMNN
    
    def printTrainErrors(self):
        '''Prints the errors of the training process'''
        if self.version != 11:
            print('\nTraining reactions error: {}'.format(self.reactTrError))
            print('Validation reactions error: {}'.format(self.reactValError))
        print('\nTraining exchanges error: {}'.format(self.exchTrError))
        print('Validation exchanges error: {}'.format(self.exchValError))

    def printTestedErrors(self, results):
        '''Print the errors of the tested dataset'''
        if self.version != 11:
            print('Test reactions error: {}'.format(results[-2]))
        print('Test exchanges error: {}'.format(results[-1]))
        
    def getErrors(self):
        '''Returns the reactions and exchange errors from the training history
        '''
        history, v = self.hist, self.version
        maeId = checkMAEVersion(history)
        if v!=11:
            react_mae=history.history['ReactLayer_{}'.format(maeId)][-1]
            val_react_mae=history.history['val_ReactLayer_{}'.format(maeId)][-1]
            out_mae=history.history['OutputLayer_{}'.format(maeId)][-1]
            val_out_mae=history.history['val_OutputLayer_{}'.format(maeId)][-1]
            return react_mae, out_mae, val_react_mae, val_out_mae
        else:
            mae = history.history[maeId][-1]
            val_mae = history.history['val_'+maeId][-1]
            return None, mae, None, val_mae

    def saveNet(self, saveFolder='savedSystems/'):
        '''Saves the MNN
        - saveFolder: str, folder to save the MNN system'''
        self.fileSaved = saveMNN(self.modelMNN, self.version, self.sparse, name = self.mainDataset.datasetName, folder = saveFolder)
        print('MNN saved in: {}'.format(self.fileSaved))
        return self.fileSaved
        
    
def doDataset(dN='ecSalm', nF=5000):
    '''Example of a dataset creation'''
    data = MNN_dataset()
    data.loadDataset(dN, nF)
    data.preprocessDataset()
    return data

