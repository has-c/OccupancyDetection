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

def predict(x, P, A, Q): #predict function
    xpred = np.matmul(A,x)
    Ppred = np.matmul(A,P)
    Ppred = np.matmul(Ppred,np.transpose(A)) + Q
    return(xpred, Ppred)

def innovation(xpred, Ppred, z, H, R): #innovation function
    nu = z - np.matmul(H,xpred)
    S = np.matmul(H,Ppred)
    S = R + np.matmul(S, np.transpose(H))
    return(nu, S)

def innovation_update(xpred, Ppred, nu, S, H):
    K = np.matmul(Ppred, np.transpose(H))
    K = np.matmul(K,np.linalg.inv(S)) #check inverse function
    xnew = xpred + np.matmul(K,nu)
    Pnew = np.matmul(K,S)
    Pnew = Ppred - np.matmul(Pnew,np.transpose(K)) 
    return(xnew, Pnew)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def data_associate(centroidPred, rthetacentroid):
    rthetacentroidCurrent = rthetacentroid
    centpredCol = np.size(centroidPred,1)
    rthetaCol = np.size(rthetacentroid,1)

    for i in list(range(0,centpredCol)):
        r1 = centroidPred[0][i]
        r2 = rthetacentroid[0]
        theta1 = centroidPred[2][i]
        theta2 = rthetacentroid[1]
        temp = np.sqrt(np.multiply(r1,r1) + np.multiply(r2,r2) - np.multiply(np.multiply(np.multiply(2,r1),r2),np.cos(theta2-theta1)))
        if(i==0):
            minDist = temp
        else:
            minDist = np.vstack((minDist,temp))

    currentFrame = np.empty((2,max(centpredCol,rthetaCol)))
    currentFrame[:] = np.nan

    minDist = np.reshape(minDist, (centpredCol,rthetaCol))
    minDistOrg = minDist

    for i in list(range(0,min(centpredCol,rthetaCol))):
        if((np.ndim(minDist)) == 1):
            minDist = np.reshape(minDist,(rthetaCol,1))
            minDistOrg = np.reshape(minDistOrg,(rthetaCol,1))
        val = np.min(minDist)
        resultOrg = np.argwhere(minDistOrg == val)
        result = np.argwhere(minDist == val)
        minRowOrg = resultOrg[0][0]
        minColOrg = resultOrg[0][1]
        minRow = result[0][0]
        minCol = result[0][1]
        currentFrame[:,minRowOrg] = rthetacentroid[:,minColOrg]
        minDist = np.delete(minDist,minRow,0)
        minDist = np.delete(minDist,minCol,1)
        rthetacentroidCurrent = np.delete(rthetacentroidCurrent,minCol,1)

    index = 0
    if (rthetacentroidCurrent.size != 0): #check indexing
        for i in list(range(centpredCol,rthetaCol)):
            currentFrame[:,i] = rthetacentroidCurrent[:,index]
            index += 1 

    return(currentFrame)

def LiveRKF(currentrawxycentroidData, centroidX, centroidP, Q, R, isFirst):
    
    
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
    if (xytransposecentroidData.size != 0): 
        [rthetacentroidData[0,:],rthetacentroidData[1,:]] = cart2pol(xytransposecentroidData[0,:],xytransposecentroidData[1,:])
    if(isFirst):
        centroidX[[0,2],0] = rthetacentroidData[[0,1],0]
        isFirst = 0
    if((rthetacentroidData.size != 0)):
        currentFrame = data_associate(centroidX, rthetacentroidData)
        addittionalCentroids = (np.size(rthetacentroidData,1)-np.size(centroidX,1))
        if(addittionalCentroids>0):
            truncateCurrentFrame = currentFrame[:,np.size(centroidX,1):np.size(currentFrame,1)]
            zeroTemplate = np.zeros((4,np.size(truncateCurrentFrame,1)),dtype=truncateCurrentFrame.dtype)
            zeroTemplate[[0,2],:] = truncateCurrentFrame[[0,1],:]
            centroidX = np.hstack((centroidX,zeroTemplate))
            for newFrameIndex in list((range(0, addittionalCentroids))):
                centroidP.extend([P])
        for currentFrameIndex in list((range(0,np.size(currentFrame,1)))):
            if(not(np.isnan(currentFrame[0,currentFrameIndex]))):
                [xpred, Ppred] = predict(centroidX[:,currentFrameIndex], centroidP[currentFrameIndex], A, Q)
                [nu, S] = innovation(xpred, Ppred, currentFrame[:, currentFrameIndex], H, R)
                [centroidX[:,currentFrameIndex],  centroidP[currentFrameIndex]] = innovation_update(xpred, Ppred, nu, S, H)
            else:
                [centroidX[:,currentFrameIndex], centroidP[currentFrameIndex]] = predict(centroidX[:,currentFrameIndex], centroidP[currentFrameIndex], A, Q)                   
    else:
        for noFrameIndex in list((range(0,np.size(centroidX,1)))):
            [centroidX[:,noFrameIndex], centroidP[noFrameIndex]] = predict(centroidX[:,noFrameIndex], centroidP[noFrameIndex], A, Q)
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

    #RKF 
    centroidX =np.zeros((4,1))
    centroidP = []
    P = np.identity(4)
    centroidP.extend([P])
    Q = np.multiply(0.2,np.identity(4))
    R = np.multiply(5,np.array([[1],[1]]))
    #tree based
    weightThresholdIntial = 0.2 #minimum distance between points
    minClusterSizeInitial = 10
    weightThresholdFinal = 0.8 #minimum distance between points
    minClusterSizeFinal = 8 

    #zone snr
    snrFirstZone = 10
    snrMiddleZone = 15
    snrLastZone = 5

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
            s2.setData(tiPosX,tiPosY)

        QtGui.QApplication.processEvents()
        
        
        
main()
