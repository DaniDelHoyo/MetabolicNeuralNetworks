'''
Author: Daniel Del Hoyo Gomez
Universidad Politécnica de Madrid
Auxiliar functions to work with the Metabolic Neural Network (MNN) objects.
'''

import cobra
import pickle
import time
import random as rd
import math
import os
import numpy as np

##########################################
#### Auxiliar functions
##########################################

def loadCobraModels(modelBases, dataFolder = 'modelsEC/'):
    '''Return a list with the cobra models specified
    Inputs:  -modelBases: list of str, names of the cobra models to load
             -dataFolder: str, name of the folder where the models are stored
    Outputs: -models: list of cobra models
    '''
    models = []
    for ib in modelBases:
        models.append(cobra.io.load_json_model(dataFolder+ib+'.json'))
    print(len(models),' models loaded')
    return models

def get_influxesIds(m):
    '''Returns a list with the metabolites that can be taken by the cell
    Input:  -m: cobra model
    Output: -influxes: list [metabolitesIds]
    '''
    influxes = []
    for exch in m.exchanges:
        if exch.reaction.split()[-1] == '<=>':
            influxes.append(exch.id)
    if 'EX_h2o_e' in influxes:
        influxes.remove('EX_h2o_e')
    return influxes

def addMetab2dic(metab, dic, ninds):
    '''Add a metabolite to the dic that was not present before, filling with zeros that metabolite in the previous indivs
    Inputs:  -metab: str, metabolite id
             -dic: dictionary with {metabsIds: inputs/outputs}
             -ninds: int, number of individuals(instances)
    Output:  -dic: dictionary with {metabsIds: inputs/outputs}
    '''
    dic[metab] = [0] * ninds
    return dic

##def random_medium(model, mini=0.0, maxi=1000):
##    '''Return a dictionary with random values for the metabolites in a medium
##    Inputs:  -model: cobramodel
##             -mini: int, minimum value for input fluxes
##             -maxi: int, maximum flux for input fluxes
##    Outputs: -medium: dictionary, {metabsIds: flux(float)}
##    '''
##    medium={}
##    inputIds = get_influxesIds(model)
##    for metab in inputIds:
##        medium[metab] = rd.uniform(mini, maxi)
##    return medium

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


def dics_to_dicOfLists(dics):
    '''Returns a merged dictionary, from lists of same ids dics to dic of lists
    Inputs:  -dics: list of dictionaries, [{metabId: flux1},{metabId: flux2}]
    Outputs: -ndic: dictionary, {metabId: [fluxes]}
    '''
    ndic = {}
    for dic in dics:
        for key in dic:
            if key in ndic:
                ndic[key].append(dic[key])
            else:
                ndic[key] = [dic[key]]
    return ndic

def findRealBiomass(exIds, exchanges):
    '''Find the real biomass index among several biomasses reactions found in the model
    checking which has the maximum values.
    Inputs:  -exIds: list, list of exchanges ids
             -outputs: numpy array, exchanges outputs
    Outputs: -bioIdx: int, index of the real biomass with non null values
    '''
    maxidx, maxis = [], []
    for i in range(len(exIds)):
        exId = exIds[i]
        if 'biomass' in exId.lower():
            maxis.append(np.mean(exchanges[:,i]))
            maxidx.append(i)
    preIdx = maxis.index(max(maxis))
    bioIdx = maxidx[preIdx]
    return bioIdx

def FBA_output_fluxes_MNN(model, medium, layers):
    '''Returns the inner reactions and exchanged+biomass fluxes with FBA
    Inputs:  -model: cobra model
             -medium: dictionary, {metabsIds: flux(float)}
             -layers: list of  2 lists with cobra reactions [reactionsCobra, exchangesCobra]
    Outputs: -fluxes: list of 2 lists with reactions fluxes [reactionsFluxes, exchangesFluxes]
    '''
    with model:
        model.medium = medium
        biomass=model.slim_optimize()
        if math.isnan(biomass):
            return None
        fluxes = []
        for lay in layers:
            fluxes.append([])
            for r in lay:
                fluxes[-1].append(r.flux)
    return fluxes

def prepareNetInput(all_inps):
    '''Returns the input prepared for the neural network
    Inputs:  -all_inps: list of lists
    Outputs: -inps: numpy array, inputs of the neural network
    '''
    inps=[]
    for ind in all_inps:
        inps.append(np.asarray(ind))
    inps = np.asarray(inps)
    return inps

def prepareNetOutput(all_outs):
    '''Returns the output prepared for the neural network
    Inputs:  -all_outs: list of lists(instances) of 2 lists with outputs: [[reactsFluxes, exchsFluxes],[reactsFluxes, exchsFluxes],...]
    Outputs: -outs: list of 2 numpy arrays [reactionsFluxes, exchangesFluxes]
    '''
    outs=[[] for i in range(len(all_outs[0]))]
    for ifba in range(len(all_outs)):
        for ilay in range(len(all_outs[ifba])):
            outs[ilay].append(np.array(all_outs[ifba][ilay]))
    for i in range(len(all_outs[0])):
        outs[i]=np.asarray(outs[i])
    return outs

def checkEqual(iterator):
    '''Check if all values in a vector are equal
    Inputs:  -iterator: iterator
    Outputs: boolean, whether all the values in the iterator are equal'''
    return len(set(iterator)) <= 1

def getInputStrain(reactions):
    '''Return the 0 / 1 np array with the absence / presence of the reactions in a strain
    Inputs:  -reactions: numpy array, outputs of the inner reactions (one strain)
    Ouputs:  -inpStrain: numpy array, inputStrain for shutdown model. Shape: (1, reactions)
    '''
    sumi = np.sum(reactions, axis=0)
    inpStrain = np.ones(shape=(1,len(sumi)))[0]
    inpStrain[np.where(abs(sumi)==0)] = 0
    return np.array([inpStrain])

def metabsNoCompartment(metabs):
    '''Return a list with the metabolites in metabs removing the compartment (metab_comp)
    Inputs:  -metabs: list of str, metabolite ids
    Outputs: -ms: list of str, metabolite ids
    '''
    ms=[]
    for m in metabs:
        ms.append('_'.join(m.id.split('_')[:-1]))
    ms=list(set(ms))
    return ms

def glayers2dic(glayers):
    '''Return a dictionary with the metabolites involved in each reaction {reactionId: [metabolites]}
    Inputs:  -glayers: list of 2 lists of cobra reactions (first list: inner reactions)
    Outputs: -rDics: list of 2 dictionaries, inner and exchange reactions ids and their metabolites {reactId:[metabs]}
    '''
    rDics = []
    for j in range(2):
        rDic = {}
        for i in range(len(glayers[j])):
            rId = glayers[j][i].id
            metabs = list(glayers[j][i].metabolites.keys())
            metabs = metabsNoCompartment(metabs)
            rDic[rId] = metabs
        rDics.append(rDic)
    return rDics

def renameBiomass(gLayDics):
    '''Renames the biomass key in the gLayDics to "Biomass"
    Inputs:  -gLayDics: list of 2 dictionaries, inner and exchange reactions ids and their metabolites {reactId:[metabs]}
    Outputs: -gLayDics: list of 2 dictionaries, inner and exchange reactions ids and their metabolites {reactId:[metabs]}
    '''
    gDic = {}
    for k in gLayDics[1]:
        if 'biomass' in k.lower():
            gDic['Biomass'] = gLayDics[1][k]
        else:
            gDic[k] = gLayDics[1][k]
    gLayDics[1] = gDic
    return gLayDics

##########################################
#### Main functions
##########################################

def createFBADataset(modelBases, nFBAs, doFBA = False, strainBase = '_MNNv2', modelsFolder = 'modelsEC/'):
    '''Creates a FBA dataset from cobra models
    Inputs:  -modelBases: list of str, names of cobra models
             -nFBAs: int, number of FBAs per strain
             -doFBA: boolean, run FBA to create new instances
             -strainBase: str, base of the strains FBA datasets
             -modelsFOlder: str, folder where the cobra models are stored
    Outputs: -netInps: numpy array, inputs of the neural network
             -netOuts: list of 2 numpy arrays, [reactions, exchanges]
             -pDic: dictionary, parameters of the dataset
    '''
    net_inputs_dic, net_reactions_dic, net_outputs_dic = {}, {}, {}
    reactionsIds, outputsIds = {}, {}
    models = loadCobraModels(modelBases, modelsFolder)
    for i in range(len(models)):
        mCobra = models[i]
        filebase = '{}{}'.format(modelBases[i], strainBase)
        print('Loading: ', filebase)
        glayers, glayersIds = grrann_layers(mCobra)
        
        if doFBA:
            netInps, netOuts = batch_of_FBAs_MNN(mCobra, glayers, nFBAs)
            saveInOut(netInps, netOuts, filebase, append=True)
        netInps, netOuts = loadInOut(filebase)
        
        #Ids of the reactions, caring that the biomass is the real
        glayers, glayersIds = grrann_layers(mCobra, netOuts[1])
        reactionsIds = updateDic(reactionsIds, glayersIds[0], glayers[0])
        outputsIds = updateDic(outputsIds, glayersIds[1], glayers[1])
     
        cur_inp_dic, cur_react_dic, cur_out_dic = dataset2dicMNN(netInps, netOuts, mCobra, nFBAs)
        net_inputs_dic = addDicIndivs(cur_inp_dic, net_inputs_dic)
        net_reactions_dic = addDicIndivs(cur_react_dic, net_reactions_dic)
        net_outputs_dic = addDicIndivs(cur_out_dic, net_outputs_dic)

    netOuts=[[],[]]
    netInps, mediumMetabs = dic2array(net_inputs_dic)
    netOuts[0], reaction_metabs = dic2array(net_reactions_dic)
    netOuts[1], fluxes_metabs_raw = dic2array(net_outputs_dic)
    
    glayersIds, glayers = [[],[]], [[],[]]
    glayersIds[0], glayers[0] = list(reactionsIds.keys()), list(reactionsIds.values())
    glayersIds[1], glayers[1] = list(outputsIds.keys()), list(outputsIds.values())

    gLayDics = glayers2dic(glayers)
    pDic = {'glayersIds': glayersIds, 'gLayDics': gLayDics, 'mediumMetabs': mediumMetabs}
    return netInps, netOuts, pDic

def dataset2dicMNN(inputs, netOutputs, FBAmodel, numberFBAs=-1):
    '''Returns the input and 2 outputs as dictionaries with the ids of the metabolites as keys
    Inputs:  -inputs: numpy array, inputs
             -netOutputs: list of 2 numpy arrays [reactions, exchanges]
             -FBAmodel: cobra model
             -numberFBAs: int, number of instances to use [-1] == all
    Outputs: -inps: dictionary {inputsIds: inputs}
             -reacts: dictionary {reactionsIds: reactionsOutputs}
             -exchs: dictionary {outputsIds: outputs}
    '''
    inps, reacts, exchs = {}, {}, {}
    reactions, exchanges = netOutputs
    mediumMetabs = get_influxesIds(FBAmodel)
    glayers, glayersIds = grrann_layers(FBAmodel, exchanges)
    if numberFBAs == -1:
        numberFBAs = inputs.shape[0]
    for i in range(len(mediumMetabs)):
        inps[mediumMetabs[i]] = list(inputs[:numberFBAs,i])
    for i in range(len(glayersIds[0])):
        reacts[glayersIds[0][i]] = list(reactions[:numberFBAs,i])
    for i in range(len(glayersIds[-1])):
        exchs[glayersIds[-1][i]] = list(exchanges[:numberFBAs,i])
    return inps, reacts, exchs

def addDicIndivs(cur_dic, total_dic, addNew=True):
    '''Add the data in the cur_dic into the total_dic
    Metabs in the total_dic that are not found in cur_dic are filled with 0s for this cur_dic individuals
    Metabs in the cur_dic that are not found in the total_dic are added filling with 0s for the previous individuals (addMetab2dic)

    Inputs:  -cur_dic: dictionary, {metabsIds: current inputs/outputs}
             -total_dic: dictionary, {metabsIds: total inputs/outputs}
             -addNew: boolean, add new metabolites found in cur_dic into total dic
    Outputs: -total_dic: dictionary, {metabsIds: total inputs/outputs}
    '''
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

def dic2array(dic):
    '''Return an array that is complementary to the dictionary and the keys of the dictionary
    Inputs:  -dic: dictionary, {metabsIds: inputs/outputs}
    Outputs: -data: numpy array, [inputs/outputs]
             -metabsIds: list, [metabolites Ids]
    '''
    data = []
    for metab in dic:
        data.append(np.array(dic[metab]))
    data = np.array(data).T
    metabsIds = list(dic.keys())
    return data, metabsIds

def grrann_layers(m, preExchanges=0):
    '''Given a cobra model, returns a list with the reactions of the model and another list
    with its exchanges and the BIOMASS reactions
    Inputs:  -m: cobramodel
             -preExchanges: numpy array, exchanges fluxes previous to biomass labeling
    Outputs: -reactions: list, cobra reactions of inner reactions
             -exchanges: list, cobra reactions of exchanges (with labeled biomass)
             -reIds: list, reactions ids
             -exIds: lits, exchanges ids
    '''
    reactions, exchanges = [], []
    reIds, exIds = [], []
    for reaction in m.reactions:
        if 'EX_' in reaction.id or 'BIOMASS' in reaction.id:
            exchanges.append(reaction)
            exIds.append(reaction.id)
        else:
            reactions.append(reaction)
            reIds.append(reaction.id)
    if type(preExchanges) == type(np.array([])):
        bioIdx = findRealBiomass(exIds, preExchanges)
        exIds[bioIdx] = 'Biomass'
    return [reactions, exchanges], [reIds, exIds]

def updateDic(dic, keys, values):
    '''Updates a dictionary with new {keys:values}
    Inputs:  -dic: dictionary
             -keys: list of str
             -values: list of floats
    Outputs: -dic: dictionary, {keys:values}
    '''
    for i in range(len(keys)):
        dic[keys[i]] = values[i]
    return dic

def batch_of_FBAs_MNN(model, layers, num=100):
    '''Returns two lists with the dictionaries of mediums and fluxes
    Inputs:  -model: cobra model
             -layers: list of  2 lists with cobra reactions [reactionsCobra, exchangesCobra]
             -num: int, number of FBAs to create
    Outputs: -all_inps: numpy array, inputs of the neural network
             -all_outs: list of 2 numpy arrays, outputs of the neural network
    '''
    all_mediums = []
    all_outs, all_inps = [], []
    i=0
    while i<num:
        all_mediums.append(random_medium(model))
        cur_fluxes = FBA_output_fluxes_MNN(model, all_mediums[-1], layers)
        if cur_fluxes != None:
            if i%(num/10)==0:
                print('Created {}/{}'.format(i,num))
            all_inps.append(list(all_mediums[-1].values()))
            all_outs.append(cur_fluxes)
            i+=1
    print('Created {}/{}'.format(num,num))
    all_inps, all_outs = prepareNetInput(all_inps), prepareNetOutput(all_outs)
    return all_inps, all_outs

def nonzeroidx_output(output, limit = 0.05):
    '''Find the metabolites that change for all the dataset. A metabolite is considered not to change if its values
    are 0 for more than (1-limit) proportion
    Inputs:  -output: numpy array, reactions/exchanges of the network
             -limit: float, proportion threshold of non-null instances to keep the variable
    '''
    nonzero = limit*output.shape[0]
    nonzeros_idx=[]
    for j in range(output.shape[1]):
        cur_nonzeros = len(np.where(abs(output[:,j])>0.0000001)[0])
        if cur_nonzeros > nonzero and not checkEqual(output[:,j]):
            nonzeros_idx.append(j)
    return nonzeros_idx

def shuffle_dataset(datas):
    '''Randomly shuffles the databaset inputs and outputs with the same order (their rows are shuffled)
    Inputs:  -datas: list of numpy arrays, datasets saved in list
    Outputs: -datas: list of numpy arrays, shuffled datasets saved in list 
    '''
    s = np.arange(datas[0].shape[0])
    np.random.shuffle(s)
    for i in range(len(datas)):
        datas[i] = datas[i][s,:]
    return datas

def real2normArray(realArray, maxs, mins):
    return (realArray-mins) / (maxs-mins)
    
def real2norm(reals, maxs, mins):
    '''Return the normalized values of the output (with predefined maxs and mins)
    Inputs:  -reals: numpy array, real values shape(1, variables)
             -maxs: list, maximum values
             -mins: list, minimum values
    Outputs: -numpy array, normalized values
    '''
    newPreds = []
    for i in range(reals.shape[1]):
        if maxs[i] != mins[i]:
            newPreds.append(float((reals[:,i]-mins[i]) / (maxs[i]-mins[i])))
        else:
            newPreds.append(mins[i])
    return np.array(newPreds)

def norm_outputs(outputs, incr=0):
    '''Normalize the outputs as (X-min)/(max-min).
    Inputs:  -outputs: numpy array, reactions/exchanges output
             -incr: float, proportion of increment to expand the normalization maximum and minimum 
    Outputs: -numpy array, normalized reactions/exchanges output
             -maxs: numpy array, maximum values
             -mins: numpy array, minimum values
    '''
    mins = np.min(outputs,axis=0) - np.abs(np.min(outputs, axis=0)*incr)
    maxs = np.max(outputs,axis=0) + np.abs(np.max(outputs, axis=0)*incr)
    return (outputs-mins) / (maxs-mins), maxs, mins

def norm_outputs_withZeros(outputs, incr=0):
    '''Normalize the outputs as (X-min)/(max-min).
    Inputs:  -outputs: numpy array, reactions/exchanges output
             -incr: float, proportion of increment to expand the normalization maximum and minimum 
    Outputs: -nouts: numpy array, normalized reactions/exchanges output
             -maxs: numpy array, maximum values
             -mins: numpy array, minimum values
    '''
    mins = np.min(outputs,axis=0) - np.abs(np.min(outputs, axis=0)*incr)
    maxs = np.max(outputs,axis=0) + np.abs(np.max(outputs, axis=0)*incr)
    nouts = outputs.copy()
    for j in range(outputs.shape[1]):
        if maxs[j] != mins[j]:
            nouts[:,j] = (outputs[:,j]-mins[j]) / (maxs[j]-mins[j])
    return nouts, maxs, mins

def getSparseInputStrains(reactions, nFBAs):
    '''Return all the inputStrains in a dataset, dividing each of the strains.
    Inputs:  -reactions: numpy array, outputs of the inner reactions of several strains(nStrains)
             -nFBAs: int, number of instances per strain
    Outputs: -sInpStrains: numpy array, sparse input Strains vectors (a row for each strain). Shape: (nStrains, reactions)
    '''
    sInpStrains = getInputStrain(reactions[:nFBAs,])
    for i in range(1, reactions.shape[0] // nFBAs):
        curInpStrain = getInputStrain(reactions[nFBAs*i:nFBAs*(i+1),])
        sInpStrains = np.vstack([sInpStrains, curInpStrain])
    return sInpStrains

def filterOutputs(netOuts, pDic, thres = 0.05):
    '''Return the filtered outputs and parameters.
    Filter the output metabolites, only predicted those with considerable change.
    Inputs:  -netOuts: list of 2 numpy arrays, [reactions, exchanges]
             -pDic: dictionary, parameters of the dataset
    Outputs: -netOuts: list of 2 numpy arrays, [reactions, exchanges]
             -pDic: dictionary, parameters of the dataset
    '''
    nonzerosIdx=[[],[]]
    nonzerosIdx[0] = nonzeroidx_output(netOuts[0], limit = thres)
    nonzerosIdx[1] = nonzeroidx_output(netOuts[-1], limit = thres)

    #Filter outputs and glayersIds
    for i in range(2):
        netOuts[i] = netOuts[i][:,nonzerosIdx[i]]
        pDic['glayersIds'][i] = np.array(pDic['glayersIds'][i])[nonzerosIdx[i]]
        gDic={}
        nks = np.array(list(pDic['gLayDics'][i].keys()))[nonzerosIdx[i]]
        for k in nks:
            gDic[k] = pDic['gLayDics'][i][k]
        pDic['gLayDics'][i] = gDic
    pDic['gLayDics'] = renameBiomass(pDic['gLayDics'])
    #Filter inp Strains and gLayDic
    pDic['sInpStrains'] = pDic['sInpStrains'][:,nonzerosIdx[0]]
    return netOuts, pDic

## Pickle save/load
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
    filename = datasetName + '_parameters.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(pDic, handle)

def loadParametersDic(datasetName):
    filename = datasetName + '_parameters.pickle'
    with open(filename, 'rb') as handle:
        pDic = pickle.load(handle)
    return pDic

