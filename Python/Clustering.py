import numpy as np
import pandas as pd
import time
from sklearn.cluster import DBSCAN

def readMeasurements(positionsDf):
    #find headers and frames within headers
    #each header has the structure frameNumber, X, Y
    #below each header is the frame data
    pointCloudPositions = list()
    headerFound = False
    for rowIndex in range(0, positionsDf.shape[0]):
        row = positionsDf.loc[rowIndex]
        if row[0] == 'FrameNumber' and row[1] == 'X' and row[2] == 'Y':
            if headerFound: 
                #actual data was found last frame and this frame actual data is found again
                #past frame ended so add to list
                frame.index = pd.Series(np.arange(0, frame.shape[0])) #reindex the frame
                pointCloudPositions.append(frame)
                frame = pd.DataFrame([])
            else:
                #header found
                headerFound = True
                frame = pd.DataFrame([])
            #data should be following
        elif headerFound:
            if np.isnan(np.float(row[1])) and np.isnan(np.float(row[2])):
                #empty row 
                pointCloudPositions.append(frame)
                headerFound = False
                #only time its going to be NaN if the frame is completely empty
            else:
                #actual data
                X = np.float(row[1])
                Y = np.float(row[2])
                frameNumber = int(row[0])
                if len(frame) == 0:
                    frame = pd.DataFrame({'X':X, 'Y':Y, 'FrameNumber':frameNumber}, index=range(1)) #first element of frame
                else:
                    data = pd.DataFrame({'X':X, 'Y':Y, 'FrameNumber':frameNumber}, index=range(frameNumber,frameNumber+1))
                    frame = pd.concat([frame,data])

    return pointCloudPositions
 
def ReadWriteClustering(filePath):
    #graph constraints and initialize variables
    weightThreshold = 0.2 #maximum distance between points
    minClusterSize = 30 #minimum cluster size
    centroidData = list()

    #read in data
    positionsDf = pd.read_csv(filePath, header=None)
    pointCloudPositions = readMeasurements(positionsDf)

    #to clear the information within the file
    #remove all information within the file
    # open(filePath, "w").close()
    
    for vertexDf in pointCloudPositions:
        pointsX = vertexDf['X'].values
        pointsY = vertexDf['Y'].values
        clusterDf = pd.DataFrame([], columns=['X', 'Y', 'CentroidNumber'])
        clusterDf.to_csv('ClusterData.csv', mode='a', header=True, index=False)

        if len(pointsX) >= minClusterSize:
            
            clusterer = DBSCAN(eps=0.5, min_samples=20)
            clusterer.fit(pd.DataFrame(np.transpose(np.array([pointsX,pointsY]))).values)

            if clusterer.core_sample_indices_.size > 0:
                #array that contains the x,y positions and the cluster association number
                clusters = np.array([pointsX[clusterer.core_sample_indices_],
                          pointsY[clusterer.core_sample_indices_], 
                         clusterer.labels_[clusterer.core_sample_indices_]])
                for centroidNumber in np.unique(clusters[2,:]):
                    xMean = np.mean(clusters[0,:][np.isin(clusters[2,:], centroidNumber)])
                    yMean = np.mean(clusters[1,:][np.isin(clusters[2,:], centroidNumber)])
            else:
                centroidDf = pd.DataFrame(np.expand_dims(np.array([np.NaN, np.NaN, 0]), axis=0))
                centroidDf.to_csv('ClusterData.csv', mode='a', index=False, header=None)
        else:
            centroidDf = pd.DataFrame(np.expand_dims(np.array([np.NaN, np.NaN, 0]), axis=0))
            centroidDf.to_csv('ClusterData.csv', mode='a', index=False, header=None)
            
def LiveClustering(vertexDf):
    
    #initialize constraints
    minClusterSize = 15
    
    #vertexDf is just the frame 
    pointsX = vertexDf['X'].values
    pointsY = vertexDf['Y'].values

    if len(pointsX) >= minClusterSize:

        clusterer = DBSCAN(eps=0.5, min_samples=20)
        clusterer.fit(pd.DataFrame(np.transpose(np.array([pointsX,pointsY]))).values)

        if clusterer.core_sample_indices_.size > 0:
            #array that contains the x,y positions and the cluster association number
            clusters = np.array([pointsX[clusterer.core_sample_indices_],
                      pointsY[clusterer.core_sample_indices_], 
                     clusterer.labels_[clusterer.core_sample_indices_]])
            for centroidNumber in np.unique(clusters[2,:]):
                xMean = np.mean(clusters[0,:][np.isin(clusters[2,:], centroidNumber)])
                yMean = np.mean(clusters[1,:][np.isin(clusters[2,:], centroidNumber)])


    return yMean, xMean
