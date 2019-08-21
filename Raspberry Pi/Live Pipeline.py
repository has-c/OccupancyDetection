import serial
import time
import numpy as np
import pandas as pd
#pyqtgraph -> fast plotting
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from sklearn.cluster import KMeans
import copy

def validateChecksum(recieveHeader):
    h = recieveHeader.view(dtype=np.uint16)
    a = np.array([sum(h)], dtype=np.uint32)
    b = np.array([sum(a.view(dtype=np.uint16))], dtype=np.uint16)
    CS = np.uint16(~(b))
    return CS


def readHeader(recieveHeader):
    headerContent = dict()
    index = 0
    
    headerContent['magicBytes'] = recieveHeader[index:index+8]
    index += 20
    
    headerContent['packetLength'] = recieveHeader[index:index+4].view(dtype=np.uint32)
    index += 4
        
    headerContent['frameNumber'] = recieveHeader[index:index+4].view(dtype=np.uint32)
    index += 24
    
    headerContent['numTLVs'] = recieveHeader[index:index+2].view(dtype=np.uint16)
    
    return headerContent

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
#         print('NUMBER OF POINTS ', str(int(numberOfPoints)))
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

def iterativeDfs(vertexID, edgeMatrix, startNode):

    visited = np.array([], dtype=np.int)
    dfsStack = np.array([startNode])

    while dfsStack.size > 0:
        # equivalent to stack pop function
        vertex, dfsStack = dfsStack[-1], dfsStack[:-1]
        if vertex not in visited:
            #find unvisited nodes
            unvisitedNodes = vertexID[np.logical_not(
                np.isnan(edgeMatrix[int(vertex), :]))]
            visited = np.append(visited, vertex)
            #add unvisited nodes to the stack
            dfsStack = np.append(
                dfsStack, unvisitedNodes[np.logical_not(np.isin(unvisitedNodes, visited))])

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
        edgeMatrix = np.sqrt(
            np.add(np.square(xDifference), np.square(yDifference)))

        #weight based reduction of graph/remove edges by replacing edge weight by np.NaN
        weightMask = np.logical_or(np.greater(
            edgeMatrix, weightThreshold), np.equal(edgeMatrix, 0))
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
                cluster = np.array([posX[visited], posY[visited], SNR[visited],
                                    np.repeat(centroidNumber, repeats=len(visited))])
                if associatedPoints.size == 0:
                    associatedPoints = cluster
                else:
                    associatedPoints = np.hstack((associatedPoints, cluster))
                centroidNumber += 1

    return associatedPoints

def predict(x, P, A, Q):  # predict function
    xpred = np.matmul(A, x)
    Ppred = np.matmul(A, P)
    Ppred = np.matmul(Ppred, np.transpose(A)) + Q
    return(xpred, Ppred)

def innovation(xpred, Ppred, z, H, R):  # innovation function
    nu = z - np.matmul(H, xpred)
    S = np.matmul(H, Ppred)
    S = R + np.matmul(S, np.transpose(H))
    return(nu, S)


def innovation_update(xpred, Ppred, nu, S, H):
    K = np.matmul(Ppred, np.transpose(H))
    K = np.matmul(K, np.linalg.inv(S))  # check inverse function
    xnew = xpred + np.matmul(K, nu)
    Pnew = np.matmul(K, S)
    Pnew = Ppred - np.matmul(Pnew, np.transpose(K))
    return(xnew, Pnew)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def data_associate(centroidPred, rthetacentroid):
    rthetacentroidCurrent = rthetacentroid
    centpredCol = np.size(centroidPred, 1)
    rthetaCol = np.size(rthetacentroid, 1)

    for i in list(range(0, centpredCol)):
        r1 = centroidPred[0][i]
        r2 = rthetacentroid[0]
        theta1 = centroidPred[2][i]
        theta2 = rthetacentroid[1]
        temp = np.sqrt(np.multiply(r1, r1) + np.multiply(r2, r2) -
                       np.multiply(np.multiply(np.multiply(2, r1), r2), np.cos(theta2-theta1)))
        if(i == 0):
            minDist = temp
        else:
            minDist = np.vstack((minDist, temp))

    currentFrame = np.empty((2, max(centpredCol, rthetaCol)))
    currentFrame[:] = np.nan

    minDist = np.reshape(minDist, (centpredCol, rthetaCol))
    minDistOrg = minDist

    for i in list(range(0, min(centpredCol, rthetaCol))):
        if((np.ndim(minDist)) == 1):
            minDist = np.reshape(minDist, (rthetaCol, 1))
            minDistOrg = np.reshape(minDistOrg, (rthetaCol, 1))
        val = np.min(minDist)
        resultOrg = np.argwhere(minDistOrg == val)
        result = np.argwhere(minDist == val)
        minRowOrg = resultOrg[0][0]
        minColOrg = resultOrg[0][1]
        minRow = result[0][0]
        minCol = result[0][1]
        currentFrame[:, minRowOrg] = rthetacentroid[:, minColOrg]
        minDist = np.delete(minDist, minRow, 0)
        minDist = np.delete(minDist, minCol, 1)
        rthetacentroidCurrent = np.delete(rthetacentroidCurrent, minCol, 1)

    index = 0
    if (rthetacentroidCurrent.size != 0):  # check indexing
        for i in list(range(centpredCol, rthetaCol)):
            currentFrame[:, i] = rthetacentroidCurrent[:, index]
            index += 1

    return(currentFrame)


def LiveRKF(currentrawxycentroidData, centroidX, centroidP, Q, R, isFirst):

    #initialise matrices
    delT = 0.0500
    A = np.array([[1, delT, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, delT],
                  [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    P = np.identity(4)

    xytransposecentroidData = currentrawxycentroidData
    rthetacentroidData = xytransposecentroidData
    if (xytransposecentroidData.size != 0):
        [rthetacentroidData[0, :], rthetacentroidData[1, :]] = cart2pol(
            xytransposecentroidData[0, :], xytransposecentroidData[1, :])
    if(isFirst):
        centroidX[[0, 2], 0] = rthetacentroidData[[0, 1], 0]
        isFirst = 0
    if((rthetacentroidData.size != 0)):
        currentFrame = data_associate(centroidX, rthetacentroidData)
        addittionalCentroids = (
            np.size(rthetacentroidData, 1)-np.size(centroidX, 1))
        if(addittionalCentroids > 0):
            truncateCurrentFrame = currentFrame[:, np.size(
                centroidX, 1):np.size(currentFrame, 1)]
            zeroTemplate = np.zeros(
                (4, np.size(truncateCurrentFrame, 1)), dtype=truncateCurrentFrame.dtype)
            zeroTemplate[[0, 2], :] = truncateCurrentFrame[[0, 1], :]
            centroidX = np.hstack((centroidX, zeroTemplate))
            for newFrameIndex in list((range(0, addittionalCentroids))):
                centroidP.extend([P])
        for currentFrameIndex in list((range(0, np.size(currentFrame, 1)))):
            if(not(np.isnan(currentFrame[0, currentFrameIndex]))):
                [xpred, Ppred] = predict(
                    centroidX[:, currentFrameIndex], centroidP[currentFrameIndex], A, Q)
                [nu, S] = innovation(
                    xpred, Ppred, currentFrame[:, currentFrameIndex], H, R)
                [centroidX[:, currentFrameIndex],  centroidP[currentFrameIndex]
                 ] = innovation_update(xpred, Ppred, nu, S, H)
            else:
                [centroidX[:, currentFrameIndex], centroidP[currentFrameIndex]] = predict(
                    centroidX[:, currentFrameIndex], centroidP[currentFrameIndex], A, Q)
    else:
        for noFrameIndex in list((range(0, np.size(centroidX, 1)))):
            [centroidX[:, noFrameIndex], centroidP[noFrameIndex]] = predict(
                centroidX[:, noFrameIndex], centroidP[noFrameIndex], A, Q)
    #centroidX is 4xN array that contains that centroid information for that frame
    return centroidX, centroidP, isFirst


def main():

    # Change the configuration file name
    configFileName = 'mmw_pplcount_demo_default.cfg'

    global CLIport
    global Dataport

    CLIport = {}
    Dataport = {}

    CLIport = serial.Serial('/dev/ttyACM0', 115200)
    if not(CLIport.is_open):
        CLIport.open()
    Dataport = serial.Serial('/dev/ttyACM1', 921600)
    if not(Dataport.is_open):
        Dataport.open()

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)

    #close control port
    CLIport.close()

    #initialise variables
    lostSync = False

    #valid header variables and constant
    magicBytes = np.array([2, 1, 4, 3, 6, 5, 8, 7], dtype='uint8')

    gotHeader = False

    frameHeaderLength = 52  # 52 bytes long
    tlvHeaderLengthInBytes = 8
    pointLengthInBytes = 16
    targetFrameNumber = 0
    targetLengthInBytes = 68

    #plotting
    app = QtGui.QApplication([])

    # Set the plot
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    win = pg.GraphicsWindow(title="Testing GUI")
    p1 = win.addPlot(title = "TI's Algorithim", row=1,col=0)
    p2 = win.addPlot(title = "Project 16", row=1,col=1)
    p1.setXRange(-6, 6)
    p1.setYRange(0, 6)
    p1.setLabel('left', text='Y position (m)')
    p1.setLabel('bottom', text='X position (m)')
    p2.setXRange(-6, 6)
    p2.setYRange(0, 6)
    p2.setLabel('left', text='Y position (m)')
    p2.setLabel('bottom', text='X position (m)')
    s1 = p1.plot([], [], pen=None, symbol='o')
    s2 = p2.plot([], [], pen=None, symbolBrush = (119,0,255), symbol='s', symbolSize=20)
    occupancyEstimate = win.addLabel("Occupancy  Estimate:0",row=0,col=1, size='20pt', bold=True, color='FF0000')

    #tracking variables
    centroidX = np.zeros((4, 1))
    centroidP = []
    P = np.identity(4)
    centroidP.extend([P])
    Q = np.multiply(0.2, np.identity(4))
    R = np.multiply(5, np.array([[1], [1]]))
    isFirst = 1

    #tree based
    weightThresholdIntial = 0.2  # minimum distance between points
    minClusterSizeInitial = 10
    weightThresholdFinal = 0.8  # minimum distance between points
    minClusterSizeFinal = 8

    #zone snr
    snrFirstZone = 10
    snrMiddleZone = 15
    snrLastZone = 5

    tlvHeaderLengthInBytes = 8
    pointLengthInBytes = 16
    targetLengthInBytes = 68

    while Dataport.is_open:
        #     print('In first while')
        while (not(lostSync) and Dataport.is_open):
            #check for a valid frame header
            if not(gotHeader):
                #             print('In second while')
                #in_waiting = amount of bytes in the buffer
                rawRecieveHeader = Dataport.read(frameHeaderLength)
    #             print('after raw header recieved')
                recieveHeader = np.frombuffer(rawRecieveHeader, dtype='uint8')
    #             print(recieveHeader)

            #magic byte check
            if not(np.array_equal(recieveHeader[0:8], magicBytes)):
                print('MAGIC BYTES ARE WRONG')
                lostSync = True
                break

            #valid the checksum
            CS = validateChecksum(recieveHeader)
            if (CS != 0):
                print('HEADER CHECKSUM IS WRONG')
                lostSync = True
                break

            #have a valid frame header
            headerContent = readHeader(recieveHeader)

            if (gotHeader):
                if headerContent['frameNumber'] > targetFrameNumber:
                    targetFrameNumber = headerContent['frameNumber']
                    gotHeader = False
                    print('FOUND SYNC AT FRAME NUMBER ' +
                          str(targetFrameNumber))
                else:
                    print('OLD FRAME')
                    gotHeader = False
                    lostSync = True
                    break

            dataLength = int(headerContent['packetLength'] - frameHeaderLength)
            
            if dataLength > 0:
                #read the rest of the packet
                rawData = Dataport.read(dataLength)
                data = np.frombuffer(rawData, dtype='uint8')

                pointCloud, targetDict = tlvParsing(data, tlvHeaderLengthInBytes, pointLengthInBytes, targetLengthInBytes)

                #target
                if len(targetDict) != 0:
                    targetX = targetDict['kinematicData'][0, :]
                    targetY = targetDict['kinematicData'][1, :]
                    print(targetY)
                    s1.setData(targetX, targetY)
#                    QtGui.QApplication.processEvents()
                                            
                #pointCloud
                if pointCloud.size > 0:
                    #constrain point cloud to within the effective sensor range
                    #range 1 < x < 6
                    #azimuth -50 deg to 50 deg
                    #doppler is greater than 0 to remove static objects
                    #check whether corresponding range and azimuth data are within the constraints

                    effectivePointCloud = np.array([])
                    for index in range(0, len(pointCloud[0, :])):
                        if (pointCloud[0, index] > 1 and pointCloud[0, index] < 6) and (pointCloud[1, index] > -50*np.pi/180 and pointCloud[1, index] < 50*np.pi/180) and pointCloud[3, index] > 0:
                            #concatenate columns to the new point cloud
                            if len(effectivePointCloud) == 0:
                                effectivePointCloud = np.reshape(
                                    pointCloud[:, index], (4, 1), order="F")
                            else:
                                point = np.reshape(
                                    pointCloud[:, index], (4, 1), order="F")
                                effectivePointCloud = np.hstack(
                                    (effectivePointCloud, point))

                    if len(effectivePointCloud) != 0:
                        posX = np.multiply(effectivePointCloud[0, :], np.sin(effectivePointCloud[1, :]))
                        posY = np.multiply(effectivePointCloud[0, :], np.cos(effectivePointCloud[1, :]))
                        SNR = effectivePointCloud[3, :]
#                        clusters = np.array([posX, posY, SNR])
                        clusters = TreeClustering(posX, posY, SNR, weightThresholdIntial, minClusterSizeInitial)

                        if clusters.size > 0:

                            #             row 1 - x
                            #             row 2 - y
                            #             row 3 - SNR
                            #             row 4 - cluster number

                            #             snr zone snr test
                            #             4.5 to the end -> last zone
                            #             3-4.5m -> middle zone
                            #             1-3m -> first zone
                            snrMask_LastZone = np.logical_and(np.greater(clusters[1, :], 4.5), np.greater(clusters[2, :], snrLastZone))  # zone 4.5m and greater
                            snrMask_MiddleZone = np.logical_and(np.logical_and(np.greater(clusters[1, :], 3), np.less_equal(clusters[1, :], 4.5)),np.greater(clusters[2, :], snrMiddleZone))  # zone 3-4.5m with SNR > 20
                            snrMask_FirstZone = np.logical_and(np.less_equal(clusters[1, :], 3), np.greater(clusters[2, :], snrFirstZone))
                            overallSnrMask = np.logical_or(np.logical_or(snrMask_FirstZone, snrMask_MiddleZone), snrMask_LastZone)
                            snrFilteredClusters = clusters[:, overallSnrMask]

                            if snrFilteredClusters.size > 0:

                                dbClusters = TreeClustering(snrFilteredClusters[0, :], snrFilteredClusters[1, :],
                                                                snrFilteredClusters[2, :],
                                                                weightThresholdFinal, minClusterSizeFinal)
                                if dbClusters.size > 0:
                                    #row 1 - x
                                    #row 2 - y
                                    #row 3 - cluster number
                                    k = int(max(dbClusters[3, :])) + 1
                                    points = np.transpose(
                                        np.array([dbClusters[0, :], dbClusters[1, :]]))

                                    #kmeans
                                    centroidClusterer = KMeans(
                                        n_clusters=k).fit(points)

                                    centroidData = np.array([centroidClusterer.cluster_centers_[
                                                            :, 0], centroidClusterer.cluster_centers_[:, 1]])

                                    #tracking
                                    centroidX, centroidP, isFirst = LiveRKF(
                                        centroidData, centroidX, centroidP, Q, R, isFirst)
                                    #plot
                                    #calculate x and y positions
                                    plotCentroidX = centroidX[:,np.logical_and((np.logical_and(centroidX[0,:]<6,centroidX[0,:]>1)), np.logical_and(centroidX[2,:]<2.2689,centroidX[2,:]>0.925))]
                                    xPositions = np.multiply(
                                        plotCentroidX[0, :], np.cos(plotCentroidX[2, :]))
                                    yPositions = np.multiply(
                                        plotCentroidX[0, :], np.sin(plotCentroidX[2, :]))
#
                                    
                                    s2.setData(xPositions, yPositions)
                                    numberOfTargets = len(xPositions)
                                    message = "Occupancy Estimate: " + str(numberOfTargets)
                                    win.removeItem(occupancyEstimate)
                                    occupancyEstimate = win.addLabel(message, row=0,col=1, size='20pt', bold=True, color='FF0000')
                                    QtGui.QApplication.processEvents()
                else:
                    centroidData = np.array([])
                    centroidX, centroidP, isFirst = LiveRKF(centroidData, centroidX, centroidP, Q, R, isFirst)
                    xPositions = np.multiply(centroidX[0, :], np.cos(centroidX[2, :]))
                    yPositions = np.multiply(centroidX[0, :], np.sin(centroidX[2, :]))
                    s2.setData(xPositions, yPositions)
                    QtGui.QApplication.processEvents()
                    
                    
                
        while lostSync:
            for rxIndex in range(0, 8):
                rxByte = Dataport.read(1)
                rxByte = np.frombuffer(rxByte, dtype='uint8')
                #if the byte received is not in sync with the magicBytes sequence then break and start again
                if rxByte != magicBytes[rxIndex]:
                    break

            if rxIndex == 7:  # got all the magicBytes
                lostSync = False
                #read the header frame
                rawRecieveHeaderWithoutMagicBytes = Dataport.read(
                    frameHeaderLength-len(magicBytes))
                rawRecieveHeaderWithoutMagicBytes = np.frombuffer(
                    rawRecieveHeaderWithoutMagicBytes, dtype='uint8')
                #concatenate the magic bytes onto the header without magic bytes
                recieveHeader = np.concatenate(
                    [magicBytes, rawRecieveHeaderWithoutMagicBytes], axis=0)
                gotHeader = True
                print('BACK IN SYNC')


main()
topa