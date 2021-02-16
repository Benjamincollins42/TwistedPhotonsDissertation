'''Module containing functions for the data pipeline for une in the CNNs.'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def list_labelsT(modesUsed, numBatchPerMode, batchSize, file_type):
    """
    Function used to generate training file names, label the data in them and provide ids.
    
    Args:
    file_type = e.g "Data/weakTurb"
    batchSize = number of images per file
    numBatchPerMode = the number of batch FILES for each mode to be used
    modesUsed = number of different OAM modes to be used (0 - modesUsed)
 
    Returns: arrays of
    DataList, DataLabel, ids
    """
    #Initialising lists
    DataList = []
    DataLabel = np.zeros(modesUsed*numBatchPerMode*batchSize)
     
    #Loop to get names and labels
    for i in range(modesUsed):
        DataLabel[(i*numBatchPerMode*batchSize):((i+1)*numBatchPerMode*batchSize)] = i
        for j in range(numBatchPerMode):
            DataList.append(file_type + str(i) + "Batch" + str(j) + ".npy")
    ids = np.arange(4*modesUsed*numBatchPerMode*batchSize)
    
    return DataList, DataLabel, ids


def list_labelsV(modesUsed, numValBatchPerMode, numBatchPerMode, batchSize, file_type):
    """
    Function used to generate validation file names, label the data in them and provide ids.

    Args:
    file_type = e.g "Data/weakTurb"
    batchSize = number of images per file
    numBatchPerMode = the number of batch FILES for each mode to be used
    modesUsed = number of diffrent OAM modes to be used (0 - modesUsed)

    Returns: arrays of
    DataList, DataLabel, ids
    """
    # Initialising lists
    DataList = []
    DataLabel = np.zeros(modesUsed * numValBatchPerMode * batchSize)

    # Loop to get names and labels
    for i in range(modesUsed):
        DataLabel[(i * numValBatchPerMode * batchSize):((i + 1) * numValBatchPerMode * batchSize)] = i
        for j in range(numValBatchPerMode):
            DataList.append(file_type + str(i) + "Batch" + str(numBatchPerMode - 1 - j) + ".npy")
    ids = np.arange(4 * modesUsed * numValBatchPerMode * batchSize)

    return DataList, DataLabel, ids



def load_data(ids, numBatchPerMode, dataList, dataLabel, batchSize):
    """
    A function used to load data and its label from files.
    
    Args:
    ids = array of ids for each image
    numBatchPerMode = the number of batch FILES for each mode to be used
    dataList = list of files the data is contained within
    dataLabel = labels for each id of true class
    batchSize = number of images per file
    """
    
    #Initalising batches to return
    X = np.zeros((0,256,256,1))
    Y = np.array([])
    x = np.zeros((1,256,256,1))
    
    for i in ids:
        #Sorting out indices to get the correct files and preprocessing
        q = i%4
        k = np.floor(i/4)
        ql = np.floor(k/numBatchPerMode*batchSize)
        j = np.floor((k-ql*numBatchPerMode*batchSize)/batchSize)
        insideIndex = k - (ql*numBatchPerMode + j) * batchSize
        
        #Getting the data
        Dat = np.load(dataList[int(j+ql*numBatchPerMode)], mmap_mode='r')
        x[0,:,:,0] = Dat[:,:,int(insideIndex)]
        y = ql
        
        #Preprocessing
        if(q==3):
            x = np.flip(x,1)
            x = np.flip(x,2)
        elif(q==1 or q==2):
            x = np.flip(x,(q))
            
        x = x/np.amax(x)

        #Building batches
        X = np.append(X, x, axis = 0)
        Y = np.append(Y,y)

    return np.array(X), np.array(Y)


def historyGraphs(historyFile, epochs_range):
    """
    A function to graph the training progress of a ML algorithm
    
    Args:
    historyFile = the history of the model
    epochs_range = range of training epochs to be displayed
    """

    #Data extraction
    acc = historyFile.history['acc']
    val_acc = historyFile.history['val_acc']
    loss = historyFile.history['loss']
    val_loss = historyFile.history['val_loss']

    plt.figure(figsize=(10, 6))

    #Accuracy graph
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    #Loss graph
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def batch_generator(ids, ML_batch_size, numBatchPerMode, dataList, dataLabel, batchSize):
    """
    A function used to fetch a batch of data from files for use in ML.

    Args:
    ids = array of ids for each image
    ML_batch_size = self explanatory
    numBatchPerMode = the number of batch FILES for each mode to be used
    dataList = list of files the data is contained within
    dataLabel = labels for each id of true class
    batchSize = number of images per file
    """
    batch = []
    while True:
        np.random.shuffle(ids)
        for i in ids:
            batch.append(i)
            if len(batch) == ML_batch_size:
                yield load_data(batch, numBatchPerMode, dataList, dataLabel, batchSize)
                batch = []


def batch_generatorV(ids, ML_batch_size, numBatchPerMode, dataList, dataLabel, batchSize):
    """
    A function used to fetch a batch of validation data from files for use in ML. No shuffle.

    Args:
    ids = array of ids for each image
    ML_batch_size = self explanatory
    numBatchPerMode = the number of batch FILES for each mode to be used
    dataList = list of files the data is contained within
    dataLabel = labels for each id of true class
    batchSize = number of images per file
    """
    batch = []
    while True:
        # np.random.shuffle(ids)
        for i in ids:
            batch.append(i)
            if len(batch) == ML_batch_size:
                yield load_data(batch, numBatchPerMode, dataList, dataLabel, batchSize)
                batch = []


def historySaver(historyFile, file_name):
    """
    Method to save the history of a CNN's training.
    
    Args:
    historyFile = the history of the model    
    file_name = e.g 'CNN_D&G/EarlyTestHistory.json' 
    """
    hist_df = pd.DataFrame(historyFile.history)

    # save to json:  
    hist_json_file = file_name
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)