"""
Offline Parsing 

"""
#import libraries
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.cluster import KMeans
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from sklearn.metrics.pairwise import pairwise_distances

def tlvParsing(data, tlvHeaderLengthInBytes, pointLengthInBytes, targetLengthInBytes):
    
    data = np.frombuffer(data, dtype = 'uint8')
    
    targetDict = dict()
    pointCloud = np.array([])
    index = 0
    #tlv header parsing
    tlvType = data[index:index+4].view(dtype=np.uint32)
    tlvLength = data[index+4:index+8].view(dtype=np.uint32)
    
    index += tlvHeaderLengthInBytes
    pointCloudDataLength = tlvLength - tlvHeaderLengthInBytes
    if tlvType.size > 0 and tlvType == 6: #point cloud TLV
        numberOfPoints = pointCloudDataLength/pointLengthInBytes
        if numberOfPoints > 0:
            p = data[index:index+pointCloudDataLength[0]].view(dtype=np.single)
            #form the appropriate array 
            #each point is 16 bytes - 4 bytes for each property - range, azimuth, doppler, snr
            pointCloud = np.reshape(p,(4, int(numberOfPoints)),order="F")
    
    #increment the index so it is possible to read the target list
    index += pointCloudDataLength
    #tlv header parsing
    tlvType = data[index[0]:index[0]+4].view(dtype=np.uint32)
    tlvLength = data[index[0]+4:index[0]+8].view(dtype=np.uint32)
    index += tlvHeaderLengthInBytes
    targetListDataLength = tlvLength - tlvHeaderLengthInBytes
    if tlvType.size > 0 and tlvType == 7: #target List TLV
        
        numberOfTargets = targetListDataLength/targetLengthInBytes
        TID = np.zeros((1, int(numberOfTargets[0])), dtype = np.uint32) #tracking IDs
        kinematicData = np.zeros((6, int(numberOfTargets[0])), dtype = np.single)
        errorCovariance = np.zeros((9, int(numberOfTargets[0])), dtype = np.single)
        gatingGain = np.zeros((1, int(numberOfTargets[0])), dtype = np.single)
        
        #increment the index so it is possible to read the target list
        targetIndex = 0
        while targetIndex != int(numberOfTargets[0]):
            TID[0][targetIndex] = data[index[0]:index[0]+4].view(dtype=np.uint32)
            kinematicData[:,targetIndex] = data[index[0]+4:index[0]+28].view(dtype=np.single)
            errorCovariance[:,targetIndex] = data[index[0]+28:index[0]+64].view(dtype=np.single)
            gatingGain[:,targetIndex] = data[index[0]+64:index[0]+68].view(dtype=np.single)
            index += targetLengthInBytes
            targetIndex += 1
            
        targetDict['TID'] = TID
        targetDict['kinematicData'] = kinematicData
        targetDict['errorCovariance'] = errorCovariance
        targetDict['gatingGain'] = gatingGain
    
    return pointCloud, targetDict

def parsePointCloud(pointCloud): #remove points that are not within the boundary
    
    effectivePointCloud = np.array([])
    
    for index in range(0, len(pointCloud[0,:])):
        if (pointCloud[0,index] > 1 and pointCloud[0,index] < 6) \
        and (pointCloud[1, index] > -50*np.pi/180 \
            and pointCloud[1, index] < 50*np.pi/180):

            #concatenate columns to the new point cloud
            if len(effectivePointCloud) == 0:
                effectivePointCloud = np.reshape(pointCloud[:, index], (4,1), order="F")
            else:
                point = np.reshape(pointCloud[:, index], (4,1),order="F")
                effectivePointCloud = np.hstack((effectivePointCloud, point))

    if len(effectivePointCloud) != 0:
        posX = np.multiply(effectivePointCloud[0,:], np.sin(effectivePointCloud[1,:]))
        posY = np.multiply(effectivePointCloud[0,:], np.cos(effectivePointCloud[1,:]))
        SNR  = effectivePointCloud[3,:]
    
        return posX,posY,SNR

def iterativeDfs(vertexID, edgeMatrix, startNode):
    
    visited = np.array([], dtype=np.int)
    dfsStack = np.array([startNode])

    while dfsStack.size > 0:
        vertex, dfsStack = dfsStack[-1], dfsStack[:-1] #equivalent to stack pop function
        if vertex not in visited:
            #find unvisited nodes
            unvisitedNodes = vertexID[np.logical_not(np.isnan(edgeMatrix[int(vertex), :]))]
            visited = np.append(visited, vertex)
            #add unvisited nodes to the stack
            dfsStack = np.append(dfsStack, unvisitedNodes[np.logical_not(np.isin(unvisitedNodes,visited))])
    
    return visited

def TreeClustering(posX, posY, SNR, weightThreshold, minClusterSize):
    
    vertexID = np.arange(len(posX))
    vertexList = np.arange(len(posX))

    associatedPoints = np.array([])

    if len(posX) >= minClusterSize:
        edgeMatrix = np.zeros((len(posX), len(posY)))

        #create distance matrix
        #x1 - x0
        xDifference = np.subtract(np.repeat(posX, repeats=len(posX)).reshape(len(posX), len(posX)), 
                                  np.transpose(np.repeat(posX, repeats=len(posX)).reshape(len(posX), len(posX))))
        #y1 - y0
        yDifference = np.subtract(np.repeat(posY, repeats=len(posY)).reshape(len(posY), len(posY)), 
                                  np.transpose(np.repeat(posY, repeats=len(posY)).reshape(len(posY), len(posY))))
        #euclidean distance calculation
        edgeMatrix = np.sqrt(np.add(np.square(xDifference), np.square(yDifference)))

        #weight based reduction of graph/remove edges by replacing edge weight by np.NaN
        weightMask = np.logical_or(np.greater(edgeMatrix,weightThreshold), np.equal(edgeMatrix, 0))
        edgeMatrix[weightMask] = np.NaN

        #perform iterative dfs
        associatedPoints = np.array([])
        
        
        centroidNumber = 0
        while vertexID.size > 0:
            startNode = vertexID[0]
            visited = iterativeDfs(vertexList, edgeMatrix, startNode)
            #remove visited nodes (ie only slice off all unvisited nodes)
            vertexID = vertexID[np.logical_not(np.isin(vertexID, visited))]
#             #visited is a component, extract cluster from it if possible
            if visited.size >= minClusterSize:
                cluster =  np.array([posX[visited], posY[visited],SNR[visited],
                                     np.repeat(centroidNumber, repeats=len(visited))])
                if associatedPoints.size == 0:
                    associatedPoints = cluster
                else:
                    associatedPoints = np.hstack((associatedPoints, cluster))
                centroidNumber += 1

    return associatedPoints

#Functions adapted from Ian Reid's Estimation II: Discrete Kalman Filter (In Compendium)

def kalmanPredictionStep(stateVariables, covarianceMatrix, systemMatrix, systemCovariance): #predict function
    predictionState = np.matmul(systemMatrix,stateVariables) #Predict usng system matrix and system variables
    predictionCovariance = np.matmul(systemMatrix,covarianceMatrix) #Error covariance prediction
    predictionCovariance = np.matmul(predictionCovariance,np.transpose(systemMatrix)) + systemCovariance
    return(predictionState, predictionCovariance)


def kalmanInnovationStep(predictionState, predictionCovariance, newMeasurement, outputMatrix, measurementCovariance): #innovation function (splits update function in two essentially)
    innovationDifference = newMeasurement - np.matmul(outputMatrix,predictionState) #difference between measured and prediction
    innovationOutput = np.matmul(outputMatrix,predictionCovariance) #innovation covariance computation
    innovationOutput = measurementCovariance + np.matmul(innovationOutput, np.transpose(outputMatrix))
    return(innovationDifference, innovationOutput)

def kalmanInnovationUpdate(predictionState, predictionCovariance, innovationDifference,innovationOutput, outputMatrix): #kalman update funciton
    kalmanGain = np.matmul(predictionCovariance, np.transpose(outputMatrix))
    kalmanGain = np.matmul(kalmanGain,np.linalg.inv(innovationOutput)) #Recurisve computation of new kalman gain
    newStatePrediction = predictionState + np.matmul(kalmanGain,innovationDifference)
    newPredictionCovariance = np.matmul(kalmanGain,innovationOutput) #Calculate new error covairance matrix
    newPredictionCovariance = predictionCovariance - np.matmul(newPredictionCovariance,np.transpose(kalmanGain)) 
    return(newStatePrediction, newPredictionCovariance)

def cart2pol(x, y): #converts cartesian to polar cooridnates 
    rho = np.sqrt(x**2 + y**2) #radial component
    phi = np.arctan2(y, x)#theta component
    return(rho, phi)

def data_associate(centroidPred, rthetacentroid): #inputs: new measurement and previous measurement
    #initialise temp arrays
    rthetacentroidCurrent = rthetacentroid 
    centpredCol = np.size(centroidPred,1)
    rthetaCol = np.size(rthetacentroid,1)

    for i in list(range(0,centpredCol)):
        r1 = centroidPred[0][i] #extract preivous radial measurement for each centroid per loop
        r2 = rthetacentroid[0]#extract all new radial measurements
        theta1 = centroidPred[2][i]
        theta2 = rthetacentroid[1]
        #calculate euclidian distance between each previous measurement and all new measurements
        temp = np.sqrt(np.multiply(r1,r1) + np.multiply(r2,r2) - np.multiply(np.multiply(np.multiply(2,r1),r2),np.cos(theta2-theta1)))
        if(i==0):
            minDist = temp
        else:
            minDist = np.vstack((minDist,temp)) #store distance matrix 

    currentFrame = np.empty((2,max(centpredCol,rthetaCol))) #initialise frame for current frame's centroids
    currentFrame[:] = np.nan

    minDist = np.reshape(minDist, (centpredCol,rthetaCol))
    minDistOrg = minDist #store distance matrix in an array for reference as minDist will be modified as associated

    for i in list(range(0,min(centpredCol,rthetaCol))): #loop through the minimum number of centroids using GNN approach
        if((np.ndim(minDist)) == 1):
            minDist = np.reshape(minDist,(rthetaCol,1))
            minDistOrg = np.reshape(minDistOrg,(rthetaCol,1))
        val = np.min(minDist) #extract smallest distance
        resultOrg = np.argwhere(minDistOrg == val) #find original indicies of minimum distance
        result = np.argwhere(minDist == val) #find new indicies of minimum distance in minDist
        minRowOrg = resultOrg[0][0] #extract original and new distance matrix indicies
        minColOrg = resultOrg[0][1]
        minRow = result[0][0]
        minCol = result[0][1]
        currentFrame[:,minRowOrg] = rthetacentroid[:,minColOrg] #extract centroid associated with minimum distnace
        minDist = np.delete(minDist,minRow,0)  #delete from the modified minimum distance so it is not associated again
        minDist = np.delete(minDist,minCol,1)
        rthetacentroidCurrent = np.delete(rthetacentroidCurrent,minCol,1)

    index = 0
    if (rthetacentroidCurrent.size != 0): #Check if centroids left unassociated
        for i in list(range(centpredCol,rthetaCol)):
            currentFrame[:,i] = rthetacentroidCurrent[:,index]#Add to new centriods (unnasociated)
            index += 1 

    return(currentFrame)

def LiveRKF(currentrawxycentroidData, centroidX, centroidP, Q, R, isFirst):
    #centroidX is 4xN array that contains that centroid information for that frame
    #currentrawxycentroidData:new measured data
    #centroidP : error covariance amtrix
    
    #initialise matrices 
    delT = 0.0500
    A = np.array([[1,delT,0,0], 
                  [0,1  ,0,0], 
                  [0,0,1,delT], 
                  [0,0,0,1]])
    H = np.array([[1,0,0,0],
                  [0,0,1,0]])
    P = np.identity(4)

    xytransposecentroidData = currentrawxycentroidData
    rthetacentroidData=xytransposecentroidData
    if (xytransposecentroidData.size != 0): #convert from cartesian to polar coordinates 
        [rthetacentroidData[0,:],rthetacentroidData[1,:]] = cart2pol(xytransposecentroidData[0,:],xytransposecentroidData[1,:])
    if(isFirst):#Initialise first centroid at its measured location
        centroidX[[0,2],0] = rthetacentroidData[[0,1],0]
        isFirst = 0#set boolean to false 
    if((rthetacentroidData.size != 0)):#if there are meausred centroids in current frame
        currentFrame = data_associate(centroidX, rthetacentroidData) #Data Association performed
        addittionalCentroids = (np.size(rthetacentroidData,1)-np.size(centroidX,1)) #How many new centroids/occupants
        if(addittionalCentroids>0):  #If new centroids: Create new matrices/columns in centriods matrix, covariance matrix etc
            truncateCurrentFrame = currentFrame[:,np.size(centroidX,1):np.size(currentFrame,1)]
            zeroTemplate = np.zeros((4,np.size(truncateCurrentFrame,1)),dtype=truncateCurrentFrame.dtype)
            zeroTemplate[[0,2],:] = truncateCurrentFrame[[0,1],:]
            centroidX = np.hstack((centroidX,zeroTemplate))#create new column for new centroids
            for newFrameIndex in list((range(0, addittionalCentroids))):
                centroidP.extend([P])#create new covariance matrix
        for currentFrameIndex in list((range(0,np.size(currentFrame,1)))):#loop through current frame of centroids
            if(not(np.isnan(currentFrame[0,currentFrameIndex]))): #if not empty
                #step1: Kalman prediction
                [predictionState, predictionCovariance] = kalmanPredictionStep(centroidX[:,currentFrameIndex], centroidP[currentFrameIndex], A, Q)
                #Kalman innovation
                [innovationDifference, innovationOutput] = kalmanInnovationStep(predictionState, predictionCovariance, currentFrame[:, currentFrameIndex], H, R)
                #Kalman update 
                [centroidX[:,currentFrameIndex],  centroidP[currentFrameIndex]] = kalmanInnovationUpdate(predictionState, predictionCovariance, innovationDifference, innovationOutput, H)
            else:#if new meausred frame has no data
                #predict using preious measurements
                [centroidX[:,currentFrameIndex], centroidP[currentFrameIndex]] = kalmanPredictionStep(centroidX[:,currentFrameIndex], centroidP[currentFrameIndex], A, Q)                   
    else:#if new measured frame has no data
        for noFrameIndex in list((range(0,np.size(centroidX,1)))): #Only kalman predict step
            [centroidX[:,noFrameIndex], centroidP[noFrameIndex]] = kalmanPredictionStep(centroidX[:,noFrameIndex], centroidP[noFrameIndex], A, Q)
    #centroidX is 4xN array that contains that centroid information for that frame
    return centroidX, centroidP,isFirst


def main():
    
    #set up plottig GUI
    app = QtGui.QApplication([])
    pg.setConfigOption('background','w')
    
    win = pg.GraphicsWindow(title="Occupancy Detection GUI")
    plot1 = win.addPlot()
    plot1.setXRange(-6,6)
    plot1.setYRange(0,6)
    plot1.setLabel('left',text = 'Y position (m)')
    plot1.setLabel('bottom', text= 'X position (m)')
    s1 = plot1.plot([],[],pen=None,symbol='o')
    plot2 = win.addPlot()
    plot2.setXRange(-6,6)
    plot2.setYRange(0,6)
    plot2.setLabel('left',text = 'Y position (m)')
    plot2.setLabel('bottom', text= 'X position (m)')
    s2 = plot2.plot([],[],pen=None,symbol='o')

    parsingMatFile = '/home/pi/Desktop/OccupancyDetection/Data/Experiment Data 2/3PeopleWalking.mat'
    tlvData = (loadmat(parsingMatFile))['tlvStream'][0]

    #Initialise Kalman Parameters  
    centroidX =np.zeros((4,1)) #Centroid X contains all tracked centroids/occupant
    centroidP = []#Centroid P contains 4x4 error covariance matrix of each occupant/centroid
    P = np.identity(4)#initialise first occupant
    centroidP.extend([P])
    Q = np.multiply(0.2,np.identity(4))#system covariance matrix
    R = np.multiply(5,np.array([[1],[1]])) #measurement covariance matrix
    #tree based
    weightThresholdIntial = 0.2 #minimum distance between points
    minClusterSizeInitial = 10
    weightThresholdFinal = 0.8 #minimum distance between points
    minClusterSizeFinal = 8 

    #zone snr
    snrFirstZone = 340
    snrMiddleZone = 140
    snrLastZone = 50

    tlvHeaderLengthInBytes = 8
    pointLengthInBytes = 16
    targetLengthInBytes = 68

    tiPosX = np.array([])
    tiPosY = np.array([])

    isFirst = 1

    for tlvStream in tlvData:

        #parsing
        pointCloud, targetDict = tlvParsing(tlvStream, tlvHeaderLengthInBytes, pointLengthInBytes, targetLengthInBytes)

        if pointCloud.size > 0:
            posX,posY,SNR = parsePointCloud(pointCloud) #dictionary that contains the point cloud data
            #initial noise reduction
            clusters = TreeClustering(posX, posY, SNR,weightThresholdIntial, minClusterSizeInitial)

            if clusters.size > 0:


    #             row 1 - x
    #             row 2 - y
    #             row 3 - SNR
    #             row 4 - cluster number

    #             snr zone snr test
    #             4.5 to the end -> last zone
    #             3-4.5m -> middle zone
    #             1-3m -> first zone
                snrMask_LastZone = np.logical_and(np.greater(clusters[1,:], 4.5), np.greater(clusters[2,:], snrLastZone)) #zone 4.5m and greater
                snrMask_MiddleZone = np.logical_and(np.logical_and(np.greater(clusters[1,:], 3), np.less_equal(clusters[1,:], 4.5)), 
                                                    np.greater(clusters[2,:], snrMiddleZone)) #zone 3-4.5m with SNR > 20
                snrMask_FirstZone = np.logical_and(np.less_equal(clusters[1,:], 3), np.greater(clusters[2,:], snrFirstZone))
                overallSnrMask = np.logical_or(np.logical_or(snrMask_FirstZone,snrMask_MiddleZone), snrMask_LastZone)

                snrFilteredClusters = clusters[:,overallSnrMask]

                if snrFilteredClusters.size > 0:
    #                 s2.setData(snrFilteredClusters[0,:], snrFilteredClusters[1,:])

                    dbClusters = TreeClustering(snrFilteredClusters[0,:], snrFilteredClusters[1,:], 
                                                    snrFilteredClusters[2,:], 
                                                    weightThresholdFinal, minClusterSizeFinal)
                    if dbClusters.size > 0:
                        #row 1 - x
                        #row 2 - y
                        #row 3 - cluster number
                        k = int(max(dbClusters[3,:])) + 1 
                        points = np.transpose(np.array([dbClusters[0,:], dbClusters[1,:]]))

                        #kmeans 
                        centroidClusterer = KMeans(n_clusters= k).fit(points)
                        centroidData = np.array([centroidClusterer.cluster_centers_[:,0], centroidClusterer.cluster_centers_[:,1]])

                        #tracking
                        centroidX, centroidP,isFirst = LiveRKF(centroidData, centroidX, centroidP, Q, R, isFirst)
                        #plot
                        #calculate x and y positions
                        xPositions = np.multiply(centroidX[0,:], np.cos(centroidX[2,:]))
                        yPositions = np.multiply(centroidX[0,:], np.sin(centroidX[2,:]))

                        s1.setData(xPositions, yPositions)


        if len(targetDict) != 0:
            #kinematic data object structure
            #row 0 - posX
            #row 1 - posY 
            #row 2 - velX
            #row 3 - velY
            #row 4 - accX
            #row 5 - accY
            tiPosX = targetDict['kinematicData'][0,:]
            tiPosY = targetDict['kinematicData'][1,:]
            print(tiPosY)
            s2.setData(tiPosX,tiPosY)

        QtGui.QApplication.processEvents()
        
        
        
main()
