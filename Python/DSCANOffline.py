"""
DBSCAN and its associated functions
"""

import serial
import time
import numpy as np

#pyqtgraph -> fast plotting
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

import copy
from scipy.io import loadmat
import sys
np.set_printoptions(threshold=sys.maxsize)
from sklearn.cluster import DBSCAN
import time

def main():
    
    #initialise variables
    lostSync = False

    #valid header variables and constant
    magicBytes = np.array([2,1,4,3,6,5,8,7], dtype= 'uint8')

    isMagicOk = False
    isDataOk = False
    gotHeader = False

    frameHeaderLength = 52 #52 bytes long
    tlvHeaderLengthInBytes = 8
    pointLengthInBytes = 16
    frameNumber = 1
    targetFrameNumber = 0
    targetLengthInBytes = 68
    
    app = QtGui.QApplication([])
    pg.setConfigOption('background','w')

    for tlvStream in tlvData:
        tlvStream = np.frombuffer(tlvStream, dtype = 'uint8')

        #tlv header
        index = 0
        #tlv header parsing
        tlvType = tlvStream[index:index+4].view(dtype=np.uint32)
        tlvLength = tlvStream[index+4:index+8].view(dtype=np.uint32)

        index += tlvHeaderLengthInBytes
        tlvDataLength = tlvLength - tlvHeaderLengthInBytes

        if tlvType == 6: 
            numberOfPoints = tlvDataLength/pointLengthInBytes
            p = tlvStream[index:index+tlvDataLength[0]].view(np.single)
            pointCloud = np.reshape(p,(4, int(numberOfPoints)),order="F")

            if not(pointCloud is None):
                #constrain point cloud to within the effective sensor range
                #range 1 < x < 6
                #azimuth -50 deg to 50 deg
                #check whether corresponding range and azimuth data are within the constraints

                effectivePointCloud = np.array([])
                for index in range(0, len(pointCloud[0,:])):
                    if (pointCloud[0,index] > 1 and pointCloud[0,index] < 6) and (pointCloud[1, index] > -50*np.pi/180 and pointCloud[1, index] < 50*np.pi/180):
                        #concatenate columns to the new point cloud
                        if len(effectivePointCloud) == 0:
                            effectivePointCloud = np.reshape(pointCloud[:, index], (4,1), order="F")
                        else:
                            point = np.reshape(pointCloud[:, index], (4,1),order="F")
                            effectivePointCloud = np.hstack((effectivePointCloud, point))

                if len(effectivePointCloud) != 0:
                    posX = np.multiply(effectivePointCloud[0,:], np.sin(effectivePointCloud[1,:]))
                    posY = np.multiply(effectivePointCloud[0,:], np.cos(effectivePointCloud[1,:]))

                    #create DBSCAN dataset - find a more efficient way to do this
                    dbscanDataSet = np.array([])
                    for pointIndex in range(0, len(posX)):
                        point = np.array([posX[pointIndex], posY[pointIndex]])
                        if pointIndex == 0:
                            dbscanDataSet = [point]
                        else:
                            dbscanDataSet = np.append(dbscanDataSet, [point], axis=0)

                    #run DBSCAN
                    db = DBSCAN(eps=0.25,metric='euclidean',min_samples=16).fit(dbscanDataSet)

                    core_samples_mask = np.zeros_like(db.labels_, dtype=bool) #return an array of zeros with the same shape as labels
                    core_samples_mask[db.core_sample_indices_] = True #place true where the index leads to a point which is in a cluster
                    labels = db.labels_
                    unique_labels = set(labels)
                    xy = np.array([])
                    for label in unique_labels:
                        if label == -1:
                            continue
                        class_member_mask = (labels == label) #mask all cluster members
                        if len(xy) == 0:
                            xy = dbscanDataSet[class_member_mask & core_samples_mask]
                        else:    
                            xy = np.concatenate((xy, dbscanDataSet[class_member_mask & core_samples_mask]),axis=0)
                    s1.setData(xy[:, 0],xy[:, 1])
                    QtGui.QApplication.processEvents() 
                    s.setData(posX,posY)
                    QtGui.QApplication.processEvents() 
                
                
