'''
Author: Daniel Del Hoyo Gomez
Universidad Politécnica de Madrid
Auxiliar functions to work with the Metabolic Neural Network (MNN) objects.
'''

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer, Dense, Input, Activation
from tensorflow.keras.activations import relu as relu
import keras.backend as K
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random as rd
import os

##########################################
#### Auxiliar functions
##########################################

class CustomConnected(Dense):
    '''Derivation of a Dense layer where the conexions can be specified'''
    def __init__(self,units,connections,**kwargs):
        self.connections = connections  
        self.unit = units
        #initalize the original Dense with all the usual arguments   
        super(CustomConnected,self).__init__(units,**kwargs)  

    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = {'connections': self.connections,
                  'units': self.units}
        base_config = super(CustomConnected, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class maskedActivationClass(Layer):
    #Layer that masks the selected neurons so their outputs become 0
    def __init__(self, myMask, normZeros=0, preActivation = relu, **kwargs):
        self.myMask, self.normZeros = myMask, normZeros
        self.preActivation = preActivation
        self.__name__='maskedActivationClass'
        self.shape=myMask.shape
        super(maskedActivationClass, self).__init__(**kwargs)
        
    def get_shape(self):
        return self.shape
    
    def call(self, inputs):
        inputs = self.preActivation(inputs)
        output = inputs * self.myMask + self.normZeros * (1-self.myMask)
        return output
        
    def get_config(self):
        config = {'myMask': self.myMask, 
                  'normZeros': self.normZeros, 
                  'preActivation' : self.preActivation}
        base_config = super(maskedActivationClass, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))  

def saveMNN(modelMNN, v, sparse, name='', folder = 'savedSystems/'):
    '''Saves the MNN system weights in a h5 file
    Inputs:  v: int, version of the MNN system
             sparse: boolean, sparsity of the system
             name: str, name for saving the system
             folder: str, folder to save the systems
    Outputs: filename: str, filename where the system weights are saved'''
    if name!='':
        name='_{}_'.format(name)
    key = str(v)+str(sparse)
    filename = 'wMNN{}v{}.h5'.format(name, key)
    modelMNN.save_weights(folder + filename)
    return filename

def versionFromKey(key):
    '''Returns the version and the sparsity
    Inputs:  key: str, key with the version and sparsity
    Outputs: v: int, version of the MNN
             sparse: boolean, sparsity of the system
    '''
    sparse = 'True' in key
    if sparse:
        v = int(key.split('True')[0])
    else:
        v = int(key.split('False')[0])
    return v, sparse

def buildFromKey(key):
    '''Build a MNN system with the version and sparsity specified in the key
    Inputs:  key:str, key with the version and sparsity
    Outputs: modelMNN: MNN system built
             v: int, version of the built system
             sparse: boolean, sparsity of the built system
    '''
    v, sparse = versionFromKey(key)
    nreit = 1 if v==24 else 0
    
    print('Loading version {}- Sparse {}'.format(v,sparse))
    #Sparse or Full version
    if sparse:
        matCon = getConnectionsMatrix(glayers)
    else:
        matCon = []
    #MNN version: 11=Conventional, 21=GRRANN, 22=HBB, 23=HBB_Shutdown, 24=Reiterative_Shutdown
    if v==11:
        modelMNN = buildConventionalNet(medium_metabs, glayersIds)
    elif v==21:
        modelMNN = build_MNNv2_1(medium_metabs, glayersIds, nreit)
    elif v==22:
        modelMNN = build_MNNv2_2(medium_metabs, glayersIds, nreit, nneurons=128, matConnections=matCon)
    elif v==23:
        modelMNN = build_MNNv2_3(medium_metabs, glayersIds, nreit, normZeros, nneurons=128, matConnections=matCon)
    elif v==24:
        modelMNN = build_MNNv2_4(medium_metabs, glayersIds, nreit, normZeros, nneurons=128, matConnections=matCon)                
    #modelMNN.summary()
    return modelMNN, v, sparse 

def loadMNN(file):
    '''Load a MNN system form the filename
    Inputs:  file: str, filename of the system (Ex: wMNN_prep_ecSalm_v22True)
    Outputs: model: MNN system loaded
             v: int, version of the built system
             sparse: boolean, sparsity of the built system
    '''
    key = file.split('v')[1].split('.')[0]
    model, v, sparse = buildFromKey(key)
    model.load_weights('savedSystems/' + file)
    return model, v, sparse

def buildConventionalNet(mediumMetabs, glayersIds):
    '''Returns a conventional neural network with the optimized parameters

    Inputs:  -mediumMetabs: list of inputs ids [id1,id2...]
             -glayersIds: list of outputs ids lists [[id11, id12...],[id21, id22...]]
    Outputs: -model: tf network model '''
    ninp, nout = len(mediumMetabs), len(glayersIds[-1])
    model = Sequential()
    model.add(Dense(64, input_shape=(ninp,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nout))
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae'])
    return model

def build_MNNv2_1(mediumMetabs, glayersIds, nreit=0, activs = ['relu', ''], matConnections=[], reactLossW = 0.5):
    '''Build a GRRANN-like Metabolic Neural Network where where each neuron represents a reaction
    The input layer covers the fluxes in the medium, then reactions layer(s) are added, to
    predict the actual flux of this metabolites into the cell, which is the first metabolic layers
    Inputs:  -mediumMetabs: list of input ids
             -glayersIds: list of two list with outputs ids
             -nreit: int, number of reiterative layers [0]
             -activs: list of str, activations of the intermediate layers
             -matConnections: list, 0/1 matrix with absence/presence connections of the last layers
             -reactLossW: float, weight of the reactions layer for loss
    Outputs: -model: tensorflow neural network 
    '''
    ninp = len(mediumMetabs)
    if matConnections == []:
        matConnections = np.ones((len(glayersIds[0]), len(glayersIds[1])))

    #Input layer with the medium concentrations
    inputLayer = Input(shape=(ninp,), name='Input_medium')

    #Reactions layers repeated nreit times as dense layers
    outs, losses, wlosses = [], {}, {}
    mnnLayers=[Dense(len(glayersIds[0]), activation = activs[0],
                     name='ReactLayer0')(inputLayer)]
    losses['ReactLayer0'] = 'mean_squared_error'
    wlosses['ReactLayer0'] = reactLossW/(nreit+1)
    outs.append(mnnLayers[-1])
    for i in range(1,nreit+1):
        mnnLayers.append(Dense(len(glayersIds[0]), activation = activs[0],
                              name='ReactLayer'+str(i))(mnnLayers[-1]))
        losses['ReactLayer'+str(i)] = 'mean_squared_error'
        wlosses['ReactLayer'+str(i)] = reactLossW/(nreit+1)
        outs.append(mnnLayers[-1])

    #Output layer with the fluxes exchanges
    if activs[1] != '':
        outputLayer = CustomConnected(len(glayersIds[1]), matConnections, activation = activs[1],
                                      name='OutputLayer')(mnnLayers[-1])
    else:
        outputLayer = CustomConnected(len(glayersIds[1]), matConnections,
                                      name='OutputLayer')(mnnLayers[-1])
    losses['OutputLayer'] = 'mean_squared_error'
    wlosses['OutputLayer'] = 1-reactLossW
    outs.append(outputLayer)

    model = Model(inputs=inputLayer,
                  outputs=outs,
                  name="MetabolicNeuralNetv2.0")
    model.compile(optimizer='Adam', metrics=['mae'],
                  loss=losses, loss_weights = wlosses)
    return model

def build_MNNv2_2(mediumMetabs, glayersIds, nreit=0, nneurons=128, activ='relu',
                  matConnections=[], reactLossW = 0.5):
    '''Build a HBB Metabolic Neural Network 
    The input layer covers the fluxes in the medium, then extra dense layers are added, to
    predict the actual flux of this metabolites into the cell, which is the first metabolic layers
    Inputs:  -mediumMetabs: list of input ids
             -glayersIds: list of two list with outputs ids
             -nreit: int, number of reiterative layers [0]
             -nneurons: int, number of context-free layer neurons
             -activ: str, activation of the intermediate layers
             -matConnections: list, 0/1 matrix with absence/presence connections of the last layers
             -reactLossW: float, weight of the reactions layer for loss
    Outputs: -model: tensorflow neural network
    '''
    ninp = len(mediumMetabs)

    #Input layer with the medium concentrations
    inputLayer = Input(shape=(ninp,), name='Input_medium')

    #Extra Dense layers
    mnnLayers=[Dense(nneurons, activation=activ)(inputLayer)]
    for i in range(1,nreit+1):
        mnnLayers.append(Dense(nneurons, activation=activ)(mnnLayers[-1]))
    
    #Reaction layer 
    outs, losses, wlosses = [], {}, {}
    rLayer=Dense(len(glayersIds[0]), activation='relu', name='ReactLayer')(mnnLayers[-1])
    losses['ReactLayer'] = 'mean_squared_error'
    wlosses['ReactLayer'] = reactLossW
    outs.append(rLayer)

    #Output layer with the fluxes exchanges
    if matConnections == []:
        outputLayer = Dense(len(glayersIds[1]), name='OutputLayer')(rLayer)
    else:
        outputLayer = CustomConnected(len(glayersIds[1]), matConnections,
                                      name='OutputLayer')(rLayer)
    losses['OutputLayer'] = 'mean_squared_error'
    wlosses['OutputLayer'] = 1-reactLossW
    outs.append(outputLayer)

    model = Model(inputs=inputLayer,
                  outputs=outs,
                  name="MetabolicNeuralNetv2.0")
    model.compile(optimizer='Adam', metrics=['mae'],
                  loss=losses, loss_weights = wlosses)
    return model

def build_MNNv2_3(mediumMetabs, glayersIds, normZeros, nreit=0, nneurons=128, activ='relu', matConnections=[], reactLossW = 0.5):
    '''Build a HBB Shutdown Metabolic Neural Network 
    The input layer covers the fluxes in the medium and a strain input defines which reactions
    layers are shuwtdown and which are active.
    Predict the actual flux of this metabolites into the cell, which is the first metabolic layers
    '''
    ninp = len(mediumMetabs)
    nReact = len(glayersIds[0])
    
    #Input layer with the medium concentrations
    inputMedium = Input(shape=(ninp,), name='InputMedium')
    inputStrain = Input(shape=(nReact,), name='InputStrain')

    #Extra Dense layers
    mnnLayers=[Dense(nneurons, activation=activ)(inputMedium)]
    for i in range(1,nreit+1):
        mnnLayers.append(Dense(nneurons, activation=activ)(mnnLayers[-1]))
    
    #Reaction layer 
    outs, losses, wlosses = [], {}, {}
    rLayer = Dense(len(glayersIds[0]), name='ReactLayer0')(mnnLayers[-1])
    rLayer = Activation(maskedActivationClass(inputStrain, normZeros), name='ReactLayer')(rLayer)
    losses['ReactLayer'] = 'mean_squared_error'
    wlosses['ReactLayer'] = reactLossW
    outs.append(rLayer)

    #Output layer with the fluxes exchanges
    if matConnections == []:
        outputLayer = Dense(len(glayersIds[1]), name='OutputLayer')(rLayer)
    else:
        outputLayer = CustomConnected(len(glayersIds[1]), matConnections,
                                      name='OutputLayer')(rLayer)
    losses['OutputLayer'] = 'mean_squared_error'
    wlosses['OutputLayer'] = 1-reactLossW
    outs.append(outputLayer)

    model = Model(inputs=[inputMedium, inputStrain],
                  outputs=outs,
                  name="MetabolicNeuralNetv2.3")
    model.compile(optimizer='Adam', metrics=['mae'],
                  loss=losses, loss_weights = wlosses)
    return model

def build_MNNv2_4(mediumMetabs, glayersIds, normZeros, nreit=0, nneurons=128, activs = ['relu', ''], matConnections=[], reactLossW = 0.5):
    '''Build a GRRANN-like Metabolic Neural Network where where each neuron represents a reaction
    The input layer covers the fluxes in the medium, then reactions layer(s) are added, to
    predict the actual flux of this metabolites into the cell, which is the first metabolic layers
    '''
    ninp = len(mediumMetabs)
    nReact = len(glayersIds[0])
    if matConnections == []:
        matConnections = np.ones((len(glayersIds[0]), len(glayersIds[1])))

    #Input layer with the medium concentrations
    inputLayer = Input(shape=(ninp,), name='Input_medium')
    inputStrain = Input(shape=(nReact,), name='InputStrain')

    #Reactions layers repeated nreit times as dense layers
    outs, losses, wlosses = [], {}, {}
    x = Dense(len(glayersIds[0]), activation = activs[0], name='preReactLayer0')(inputLayer)
    mnnLayers=[Activation(maskedActivationClass(inputStrain, normZeros), name='ReactLayer0')(x)]
    #mnnLayers=[maskedActivationClass(inputStrain, normZeros, name='ReactLayer0')(x)]
    losses['ReactLayer0'] = 'mean_squared_error'
    wlosses['ReactLayer0'] = reactLossW/(nreit+1)
    outs.append(mnnLayers[-1])
    for i in range(1,nreit+1):
        x = Dense(nneurons, activation = 'relu', name='sandwichLayer')(mnnLayers[-1])
        x = Dense(len(glayersIds[0]), activation = activs[0], name='preReactLayer'+str(i))(x)
        mnnLayers.append(Activation(maskedActivationClass(inputStrain, normZeros), name='ReactLayer')(x))
        losses['ReactLayer'] = 'mean_squared_error'
        wlosses['ReactLayer'] = reactLossW/(nreit+1)
        outs.append(mnnLayers[-1])

    #Output layer with the fluxes exchanges
    if activs[1] != '':
        outputLayer = CustomConnected(len(glayersIds[1]), matConnections, activation = activs[1],
                                      name='OutputLayer')(mnnLayers[-1])
    else:
        outputLayer = CustomConnected(len(glayersIds[1]), matConnections,
                                      name='OutputLayer')(mnnLayers[-1])
    losses['OutputLayer'] = 'mean_squared_error'
    wlosses['OutputLayer'] = 1-reactLossW
    outs.append(outputLayer)

    model = Model(inputs=[inputLayer, inputStrain],
                  outputs=outs,
                  name="MetabolicNeuralNetv2.0")
    model.compile(optimizer='Adam', metrics=['mae'],
                  loss=losses, loss_weights = wlosses)
    return model

def divide_set(data, trainProp, valProp):
    '''Return the set divided in training, validationa and test sets
    Inputs:  -data: list of numpy arrays, data to divide
             -trainProp: float, proportion of the training set
             -valProp: float, proportion of the validation set (rest is the testing proportion)
    Outputs: -xTrain: list of numpy arrays, training dataset
             -xVal: list of numpy arrays, validation dataset
             -xTest: list of numpy arrays, testing dataset
    '''
    valProp = trainProp + valProp
    xTrain, xVal, xTest = [], [], []
    for i in range(len(data)):
        xTrain.append(data[i][:int(trainProp * data[i].shape[0])])
        xVal.append(data[i][int(trainProp * data[i].shape[0]) : int(valProp * data[i].shape[0])])
        xTest.append(data[i][int(valProp * data[i].shape[0]):])
    return xTrain, xVal, xTest

def reitOutputs(netOuts, nreit):
    '''Repeats the first layer of the outputs the number of times that is repeated in the model
    Inputs:  -netOuts: list of numpy arrays, output dataset
             -nreit: int, number of reiterative layers
    Outputs: -nOuts: list of numpy arrays, output dataset with the reiterative layers
    '''
    nOuts=[netOuts[0]]
    for i in range(nreit):
        nOuts.append(netOuts[0])
    nOuts.append(netOuts[1])
    return nOuts

def checkMAEVersion(h):
    '''Check if the mae keys are written as "mae" or "mean_absolute_error" and return the string
    Inputs:  -h: history of a trained tensorflow network
    Outputs: -maeId: str, id used for mae in the history
    '''
    maeId = 'mae'
    for k in h.history.keys():
        if 'mean_absolute_error' in k:
            maeId = 'mean_absolute_error'
    return maeId
    
def getConnectionsMatrix(gLayDics):
    '''Build the sparse connections matrix for the last layer of the MNN
    Reactions in previous layer which use or produce the metabolite exchanged in the neuron
    of the output layer are set to 1.
    Inputs:  -gLayDics: list of 2 dictionaries, inner and exchange reactions ids and their metabolites {reactId:[metabs]}
    Outputs: -mat: numpy array, matrix with absence/presence 0/1 of connections in the last layer
                   shape: (reactions, exchanges)
    '''
    reactions, exchanges = gLayDics
    mat = np.zeros((len(reactions), len(exchanges)))
    for i in range(len(reactions)):
        metabs = reactions[list(reactions.keys())[i]]
        for j in range(len(exchanges)):
            exchangeMetabs = exchanges[list(exchanges.keys())[j]]
            common = set(metabs).intersection(set(exchangeMetabs))
            if len(common)>0:
                mat[i,j]=1
    return mat
    
def randomInstances(number, v, inputIds, sInpStrains=[], strain=0):
    '''Returns a set of random input instances ready to be used by the MNN
    Inputs:  -number: int, number of instances
             -v: int, version of the MNN
             -inputIds: list of str, ids of the inputs
             -sInputStrains: list of lists, inputs specific of each strain for the shutdown method
             -strain: int, idx of the input strain
    Outputs: -xTest: list of numpy arrays, inputs ready for the MNN
    '''
    rds = []
    for i in range(number):
        rds.append(list(random_medium(inputIds, isModel=False).values()))
    xTest = []
    if v in [23, 24]:
        for r in rds:
            inpS = np.array([sInpStrains[strain] for i in range(number)])
            xTest = [np.array(rds), inpS]
    else:
        xTest = [rds]
    return xTest

def random_medium(inputIds, isModel=True, mini=0.0, maxi=1000):
    '''Return a dictionary with random values for the metabolites in a medium
    Inputs:  -inputIds: list of str, names of the inputs
                    or  cobramodel (if isModel==True)
             -mini: int, minimum value for input fluxes
             -maxi: int, maximum flux for input fluxes
    Outputs: -medium: dictionary, {metabsIds: flux(float)}
    '''
    medium={}
    if isModel:
        inputIds = get_influxesIds(inputIds)
    for metab in inputIds:
        medium[metab] = rd.uniform(mini, maxi)
    return medium

def prepareInputs(inputs, v, pDic):
    if v in [23, 24]:
        inpStrains = getTotalInputStrains(pDic['sInpStrains'], pDic['nFBAs'])
        inputs = [inputs, inpStrains]
    else:
        inputs = [inputs]
    return inputs


##########################################
#### Main functions
##########################################

def evaluateModel(model, xTest, yTest):
    results = model.evaluate(xTest, yTest, verbose=2)
    return results

def getTotalInputStrains(sInpStrains, nFBAs):
    '''Return the inpStrains for the MNN, with repeated rows of sInputStrains.
    Inputs:  -sInpStrains: numpy array, sparse input Strains vectors (a row for each strain). Shape: (nStrains, reactions)
             -nFBAs: int, number of instances per strain
    Outputs: -inpStrains: numpy array, inpStrain for shutdown model. Shape: (instances, reactions)
    '''
    inpStrains = np.repeat(sInpStrains, [nFBAs for i in range(sInpStrains.shape[0])], axis=0)
    return inpStrains

def buildNetwork(v, mediumMetabs, glayersIds, normZeros, nreit, matCon=[]):
    '''Build the neural network depending on the version
    Inputs:  -v: int, MNN version: 11=Conventional, 21=GRRANN, 22=HBB, 23=HBB_Shutdown, 24=Reiterative_Shutdown
             -mediumMetabs: list, ids of the inputs
             -glayersIds: list of 2 lists, ids of the outputs
    Outputs: -modelMNN: tensorflow neural network 
    '''
    matCon = []
    if v==11:
        modelMNN = buildConventionalNet(mediumMetabs, glayersIds)
    if v==21:
        modelMNN = build_MNNv2_1(mediumMetabs, glayersIds, nreit)
    elif v==22:
        modelMNN = build_MNNv2_2(mediumMetabs, glayersIds, nreit, nneurons=128,
                                 matConnections=matCon)
    elif v==23:
        modelMNN = build_MNNv2_3(mediumMetabs, glayersIds, normZeros, nreit, nneurons=128,
                                 matConnections=matCon)
    elif v==24:
        modelMNN = build_MNNv2_4(mediumMetabs, glayersIds, normZeros, nreit, nneurons=128,
                                 matConnections=matCon)

    modelMNN.summary()
    return modelMNN

def divideSetsVersion(v, netInps, netOuts, inpStrains, trainProp, valProp):
    '''Generate the training, validation and testing datasets according to the network version
    Inputs:  -v: int, MNN version: 11=Conventional, 21=GRRANN, 22=HBB, 23=HBB_Shutdown, 24=Reiterative_Shutdown
             -netInps: numpy array, inputs dataset
             -netOuts: list of 2 numpy arrays, outputs dataset
             -trainProp: float, proportion of the training set
             -valProp: float, proportion of the validation set (rest is the testing proportion)
             -inpStrains: numpy array, 0/1 absence/presence of inner reactions inputs
    Outputs: -xTrain / yTrain: list of numpy arrays, training dataset input / output
             -xVal / yVal: list of numpy arrays, validation dataset input / output
             -xTest / yTest: list of numpy arrays, testing dataset input / output
    '''
    #print('\nTraining dataset shape')
    if v==11:
        xTrain, xVal, xTest = divide_set([netInps], trainProp, valProp)
        yTrain, yVal, yTest = divide_set([netOuts], trainProp, valProp)
        xTrain, yTrain = xTrain[0], yTrain[0]
        
##        print('Input shape: ',np.array(xTrain).shape)
##        print('Exchanges shape: ',np.array(yTrain).shape)
    elif v in [23, 24]:
        xTrain, xVal, xTest = divide_set([netInps, inpStrains], trainProp, valProp)
        yTrain, yVal, yTest = divide_set(netOuts, trainProp, valProp)
        
##        print('Input shape: ',np.array(xTrain[0]).shape)
##        print('Shutdown shape: ',np.array(xTrain[1]).shape)
##        print('Reactions shape: ',np.array(yTrain[0]).shape)
##        print('Exchanges shape: ',np.array(yTrain[-1]).shape) 
    else:
        xTrain, xVal, xTest = divide_set([netInps], trainProp, valProp)
        yTrain, yVal, yTest = divide_set(netOuts, trainProp, valProp)
        xTrain = xTrain[0]
        
##        print('Input shape: ',np.array(xTrain).shape)
##        print('Reactions shape: ',np.array(yTrain[0]).shape)
##        print('Exchanges shape: ',np.array(yTrain[-1]).shape)
##    print()
    return [xTrain, xVal, xTest], [yTrain, yVal, yTest]

def adjustOutput(v, oriNetOuts, nreit=0):
    '''Adjust the network output according to the network version to use
    Inputs:  -v: int, MNN version: 11=Conventional, 21=GRRANN, 22=HBB, 23=HBB_Shutdown, 24=Reiterative_Shutdown
             -oriNetOuts: list of 2 numpy arrays, outputs dataset
             -nreit: int, number of reiterative layers
    Outputs: -netOuts: list of 2 numpy arrays, outputs dataset
    '''
    if v==24:
        nreit=1  
    if v==11:
        netOuts = oriNetOuts[1]
    elif v in [21, 24]:
        netOuts = reitOutputs(oriNetOuts, nreit)
    else:
        netOuts = oriNetOuts
    return netOuts, nreit

def shuffle_dataset(datas):
    '''Randomly shuffles the databaset inputs and outputs with the same order (their rows are shuffled)
    Inputs:  -datas: list of numpy arrays, datasets saved in list
    Outputs: -datas: list of numpy arrays, shuffled datasets saved in list 
    '''
    s = np.arange(datas[0].shape[0])
    np.random.shuffle(s)
    for i in range(len(datas)):
        datas[i] = np.array(datas[i])
        datas[i] = datas[i][s,:]
    return datas

def makeEmptyDics(pDic):
    inpDic, reacDic, exchDic = {}, {}, {}
    for k in pDic['mediumMetabs']:
        inpDic[k]=[]
    for k in pDic['glayersIds'][0]:
        reacDic[k]=[]
    for k in pDic['glayersIds'][1]:
        exchDic[k]=[]
    return inpDic, reacDic, exchDic

def addMetab2dic(metab, dic, ninds):
    '''Add a metabolite to the dic that was not present before, filling with zeros that metabolite in the previous indivs'''
    dic[metab] = [0] * ninds
    return dic

def addDicIndivs(cur_dic, total_dic, addNew=True):
    '''Add the data in the cur_dic into the total_dic
    Metabs in the toal_dic that are not found in cur_dic are filled with 0s for this cur_dic individuals
    Metabs in the cur_dic that are not found in the total_dic are added filling with 0s for the previous individuals (addMetab2dic)'''
    ninds = len(cur_dic[list(cur_dic.keys())[0]])
    if total_dic == {}:
        prev_ninds = 0
    else:
        prev_ninds = len(total_dic[list(total_dic.keys())[0]])
    for metab in total_dic:
        if metab in cur_dic:
            total_dic[metab] += cur_dic[metab]
        else:
            total_dic[metab] += [0]*ninds
    if addNew:
        for metab in cur_dic:
            if not metab in total_dic:
                total_dic = addMetab2dic(metab,total_dic, prev_ninds)
                total_dic[metab] += cur_dic[metab]            
    return total_dic

def dataset2dic(inputs, netOutputs, pDic, numberFBAs=-1):
        '''Returns the input, output as dictionaries with the ids of the metabolites as keys
        Inputs:  -inputs: array, inputs with the medium metabolites
                 -netOutputs: list of 2 arrays with reactions and exchanges outputs
                 -pDic: dictionary with the parameters of the dataset
                 -numberFBAs: number of FBA intances to use (-1:all)
        Outputs: -inps: dictionary, mediumMetabs as keys, input fluxes as values
                 -reacts: dictionary, glayerIds[0] as keys, reactions fluxes as values
                 -outs: dictionary, glayerIds[1] as keys, exchanges fluxes as values
        '''
        inps, reacts, exchs = {}, {}, {}
        reactions, exchanges = netOutputs

        if numberFBAs == -1:
            numberFBAs = inputs.shape[0]
        medium_metabs = pDic['mediumMetabs']
        glayersIds = pDic['glayersIds']
            
        for i in range(len(medium_metabs)):
            inps[medium_metabs[i]] = list(inputs[:numberFBAs,i])
        for i in range(len(glayersIds[0])):
            reacts[glayersIds[0][i]] = list(reactions[:numberFBAs,i])
        for i in range(len(glayersIds[-1])):
            exchs[glayersIds[-1][i]] = list(exchanges[:numberFBAs,i])
         
        return inps, reacts, exchs

def input2dic(inputs, pDic, numberFBAs):
    inps = {}
    if numberFBAs == -1:
        numberFBAs = inputs.shape[0]
    medium_metabs = pDic['mediumMetabs']
    for i in range(len(medium_metabs)):
        inps[medium_metabs[i]] = list(inputs[:numberFBAs,i])
    return inps

def adaptInputMultiStrain(newInps, new_pDic, pDic):
    inputs_dic, reactions_dic, exchanges_dic = makeEmptyDics(pDic)
    cur_inp_dic = input2dic(newInps, new_pDic, pDic['nFBAs'])

    new_inputs_dic = addDicIndivs(cur_inp_dic, inputs_dic, False)
    newInps = dic2array(new_inputs_dic)[0]
    return newInps
    

def dic2array(dic):
    '''Return an array that is complementary to the dictionary and the keys of the dictionary'''
    data = []
    for metab in dic:
        data.append(np.array(dic[metab]))
    data = np.array(data).T
    return data, list(dic.keys())

def adaptInputOutputMultiStrain(newInps, newOuts, new_pDic, pDic):
    '''Adapts a FBAmodel dataset to have the same inputs and outpus of the neural network trained model
    Modify the dataset for having the same input-output metabolites as the net expectes (0 if they don't exist)
    Inputs:  -newInps, newOuts: inputs and outputs of the new dataset
             -pDic: parameters of the MNN dataset
    Outputs: -new_*_dic: inputs and outputs of the new dataset with the MNN structure'''
    #Remove mold individuals from the dataset, adapting the dataset to the model inputs, outputs
    inputs_dic, reactions_dic, exchanges_dic = makeEmptyDics(pDic)
    cur_inp_dic, cur_react_dic, cur_exch_dic = dataset2dic(newInps, newOuts, new_pDic)
        
    new_inputs_dic = addDicIndivs(cur_inp_dic, inputs_dic, False)
    new_reactions_dic = addDicIndivs(cur_react_dic, reactions_dic, False)
    new_exchanges_dic = addDicIndivs(cur_exch_dic, exchanges_dic, False)

    newInps = dic2array(new_inputs_dic)[0]
    newOuts = [dic2array(new_reactions_dic)[0], dic2array(new_exchanges_dic)[0]]

    return newInps, newOuts

####################
#Save/load functions
####################

def saveInOut(netInps, netOuts, filebase, append=True):
    '''Save the input and output using pickle.
    Inputs:  -filebase: str, base of the filenames
             -netInps: numpy array, inputs of the neural network
             -netOuts: list of 2 numpy arrays, [reactions, exchanges]
             -append: boolean, whether to append the new data if the file exists
    '''
    infn, outfn = filebase + '_inputs.pickle', filebase + '_outputs.pickle'
    if infn in os.listdir() or outfn in os.listdir():
        if append:
            preinp, preout = loadInOut(filebase)
            netInps = np.append(preinp,netInps,axis=0)
            for i in range(len(preout)):
                netOuts[i] = np.append(preout[i],netOuts[i],axis=0)
    with open(infn, 'wb') as handle:
        pickle.dump(netInps, handle)
    with open(outfn, 'wb') as handle:
        pickle.dump(netOuts, handle)
    return

def loadInOut(filebase):
    '''Load the input and output using pickle
    Inputs:  -filebase: str, base of the filenames
    Outputs: -netInps: numpy array, inputs of the neural network
             -netOuts: list of 2 numpy arrays, [reactions, exchanges]
    '''
    infn, outfn = filebase + '_inputs.pickle', filebase + '_outputs.pickle'
    with open(infn,'rb') as finputs:
        netInps = pickle.load(finputs)
    with open(outfn, 'rb') as foutputs:
        netOuts = pickle.load(foutputs)
    return netInps, netOuts

def saveParametersDic(pDic, datasetName):
    '''Save the parameters dictionary in a pickle file
    Inputs:  pDic: dictionary, parameters of the dataset
             datasetName: str, name of the dictionary's dataset
    '''
    filename = datasetName + '_parameters.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(pDic, handle)

def loadParametersDic(datasetName):
    '''Load the parameters dictionary from a pickle file
    Inputs:  datasetName: str, name of the dictionary's dataset
    Outputs: pDic: dictionary, parameters of the dataset    
    '''
    filename = datasetName + '_parameters.pickle'
    with open(filename, 'rb') as handle:
        pDic = pickle.load(handle)
    return pDic

###################
#Plotting functions
###################

def plot_output(v, history, nreit, which=2):
    '''Plot the results of the training of a neural network
    Inputs:  -history: history of a trained tensorflow network
             -nreit: int, number of reiterative layers
    '''
    # PLOT LOSS AND ACCURACY
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data sets for each training epoch
    #-----------------------------------------------------------
    maeId = checkMAEVersion(history)
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    if v!=11:
        react_mae=history.history['ReactLayer_{}'.format(maeId)]
        val_react_mae=history.history['val_ReactLayer_{}'.format(maeId)]
        out_mae=history.history['OutputLayer_{}'.format(maeId)]
        val_out_mae=history.history['val_OutputLayer_{}'.format(maeId)]
    else:
        mae = history.history[maeId]
        val_mae = history.history['val_'+maeId]
    epochs=range(1,len(loss)+1) # Get number of epochs

    #------------------------------------------------
    # Plot training and validation MAE per epoch
    #------------------------------------------------
    if which in [1,2]:
        if v!=11:
            plt.plot(epochs, react_mae, 'pink', label="Reactions Training MAE")
            plt.plot(epochs, out_mae, 'c', label="Output Training MAE")
            plt.plot(epochs, val_react_mae, 'r', label="Reactions Validation MAE")
            plt.plot(epochs, val_out_mae, 'b', label="Output Validation MAE")
        else:
            plt.plot(epochs, mae, 'r', label="Output Training MAE")
            plt.plot(epochs, val_mae, 'b', label="Output Validation MAE")
        plt.legend(loc='upper right')
        plt.title('Validation MAE')
        plt.grid(linestyle='dotted')
        plt.xlabel('Number of epochs')
        plt.ylabel('MAE')
        plt.show()
    
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    if which in [0,2]:
        plt.plot(epochs[1:], loss[1:], 'r', label="Loss")
        plt.plot(epochs[1:], val_loss[1:], 'b', label="Validation loss")
        plt.legend(loc='upper right')

        plt.title('Training and validation loss')
        plt.grid(linestyle='dotted')
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
        plt.show()

