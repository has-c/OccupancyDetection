import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import time
import numpy as np
import pandas as pd

def readMeasurements(file):
    #read in measurements from csv
    rawCentroidData = pd.read_csv(file, header=None)
    #find headers and frames within headers
    #each header has the structure X, Y, CentroidNumber
    #below each header is the frame data
    centroidFrames = list()
    headerFound = False
    for rowIndex in range(0, rawCentroidData.shape[0]):
        row = rawCentroidData.loc[rowIndex]
        if row[0] == 'FilteredRange': #header has been found
            if headerFound: 
                #actual data was found last frame and this frame actual data is found again
                #past frame ended so add to list
                centroidFrames.append(frame)
                frame = pd.DataFrame([])
            else:
                #header found
                headerFound = True
                frame = pd.DataFrame([])
            #data should be following
        elif headerFound:
            #actual data
            radius = np.float(row[0])
            theta = np.float(row[2])
            centroidNumber = int(row[4])
            if len(frame) == 0:
                frame = pd.DataFrame({'Range':radius, 'Theta':theta, 'CentroidNumber':centroidNumber}, index=range(1))
            else:
                data = pd.DataFrame({'Range':radius, 'Theta':theta, 'CentroidNumber':centroidNumber}, index=range(centroidNumber,centroidNumber+1))
                frame = pd.concat([frame,data])
    
    return centroidFrames

def Graphing(filePath, win, s1):
    
    centroidFrames = readMeasurements(filePath)

    #remove all information within the file
    # open(filePath, "w").close()

    #plot information
    for frame in centroidFrames:
        #plot all required information
        if np.equal(frame.values, np.array([[0,0,0]])).all():
            continue
        
        #slice out all incorrect frames
        #if 1 < R < 6 and 50 < Theta < 130 (in degrees)
        frame = frame[np.less(frame['Range'],5.5).values]
        frame = frame[np.greater(frame['Range'],1).values]
        frame = frame[np.greater(frame['Theta'], 
                                np.multiply(np.pi,
                                            np.divide(40, 180))).values]
        frame = frame[np.less(frame['Theta'], 
                            np.multiply(np.pi,
                                        np.divide(140, 180))).values]
        #find x and y positions and occupancy number
        posX = np.multiply(frame['Range'], np.cos(frame['Theta']))
        posY = np.multiply(frame['Range'], np.sin(frame['Theta']))
        numberOfTargets = len(posX)
        
        s1.setData(posX.values, posY.values)
        QtGui.QApplication.processEvents() 
    
    

    