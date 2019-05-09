import serial
import time
import numpy as np
from sklearn.cluster import DBSCAN
#pyqtgraph -> fast plotting
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

import copy

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

def tlvParsing(data, dataLength, tlvHeaderLengthInBytes, pointLengthInBytes, targetLengthInBytes):
    
    targetDict = dict()
    pointCloud = None
    index = 0
    #tlv header parsing
    tlvType = data[index:index+4].view(dtype=np.uint32)
    tlvLength = data[index+4:index+8].view(dtype=np.uint32)
    #TLV size check
    if (tlvLength + index > dataLength):
        print('TLV SIZE IS WRONG')
        lostSync = True
        return pointCloud, targetDict
    
    index += tlvHeaderLengthInBytes
    pointCloudDataLength = tlvLength - tlvHeaderLengthInBytes
    if tlvType == 6: #point cloud TLV
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
    if tlvType == 7: #target List TLV
        
        numberOfTargets = targetListDataLength/targetLengthInBytes
        TID = np.zeros((1, int(numberOfTargets[0])), dtype = np.uint32) #tracking IDs
        kinematicData = np.zeros((6, int(numberOfTargets[0])), dtype = np.single)
        errorCovariance = np.zeros((9, int(numberOfTargets[0])), dtype = np.single)
        gatingGain = np.zeros((1, int(numberOfTargets[0])), dtype = np.single)
        
        #increment the index so it is possible to read the target list
        targetIndex = 0
        while targetIndex != int(numberOfTargets):
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

def validateChecksum(recieveHeader):
    h = recieveHeader.view(dtype=np.uint16)
    a = np.array([sum(h)], dtype=np.uint32)
    b = np.array([sum(a.view(dtype=np.uint16))], dtype=np.uint16)
    CS = np.uint16(~(b))
    return CS

def main():
    
    #user macros
    useTargetInfo = False
    usePointCloud = True
    
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
    
    # Change the configuration file name

    configFileName = 'mmw_pplcount_demo_default.cfg'
    
    global lostSync
    lostSync = False
    
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
    
    #read and parse data
    #plotting
    app = QtGui.QApplication([])

    # Set the plot 
    pg.setConfigOption('background','w')
    winPointCloud = pg.GraphicsWindow(title="Point Cloud")
    windowTemp = pg.GraphicsWindow(title="Temporary Visualisation")
    winTarget = pg.GraphicsWindow(title="Target")
    p = winPointCloud.addPlot()
    t = winTarget.addPlot()
    temp = windowTemp.addPlot()
    p.setXRange(-6,6)
    p.setYRange(0,6)
    p.setLabel('left',text = 'Y position (m)')
    p.setLabel('bottom', text= 'X position (m)')
    t.setXRange(-6,6)
    t.setYRange(0,6)
    t.setLabel('left',text = 'Y position (m)')
    t.setLabel('bottom', text= 'X position (m)')
    temp.setXRange(-6,6)
    temp.setYRange(0,6)
    temp.setLabel('left',text = 'Y position (m)')
    temp.setLabel('bottom', text= 'X position (m)')
    s1 = p.plot([],[],pen=None,symbol='o') #point cloud
    s2 = t.plot([],[],pen=None,symbol='x') #target
    s3 = temp.plot([],[],pen=None,symbol='o') #temporary data

    while Dataport.is_open:
    #     print('In first while')
        while (not(lostSync) and Dataport.is_open):
            #check for a valid frame header
            if not(gotHeader):
    #             print('In second while')
                #in_waiting = amount of bytes in the buffer
                rawRecieveHeader = Dataport.read(frameHeaderLength)
    #             print('after raw header recieved')
                recieveHeader = np.frombuffer(rawRecieveHeader, dtype = 'uint8')
    #             print(recieveHeader)

            #magic byte check
            if not(np.array_equal(recieveHeader[0:8],magicBytes)):
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
                    print('FOUND SYNC AT FRAME NUMBER ' + str(targetFrameNumber))
                else:
                    print('OLD FRAME')
                    gotHeader = False
                    lostSync = True
                    break

            dataLength = int(headerContent['packetLength'] - frameHeaderLength)

            if dataLength > 0:
                #read the rest of the packet
                rawData = Dataport.read(dataLength)
                data = np.frombuffer(rawData, dtype = 'uint8')

                pointCloud, targetDict = tlvParsing(data, dataLength, tlvHeaderLengthInBytes, pointLengthInBytes,targetLengthInBytes)

                #target
                if useTargetInfo:
                    if len(targetDict) != 0:
                            targetX = targetDict['kinematicData'][0,:]
                            targetY = targetDict['kinematicData'][1,:]
                            s2.setData(targetX,targetY)
                            QtGui.QApplication.processEvents() 

                #pointCloud
                if usePointCloud:
                    if not(pointCloud is None):
                        #constrain point cloud to within the effective sensor range
                        #range 1 < x < 6
                        #azimuth -50 deg to 50 deg
                        #doppler is greater than 0 to remove static objects
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
                                    
                            if len(xy) == 0:
                                s3.setData([],[])
                                QtGui.QApplication.processEvents()
                                print('PEOPLE COUNT: 0')
                            else:
                                s3.setData(xy[:, 0],xy[:, 1])
                                QtGui.QApplication.processEvents()
                                print('PEOPLE COUNT: ', str(max(unique_labels)+1))
                                s1.setData(posX,posY)
                                QtGui.QApplication.processEvents() 
                            
                            



        while lostSync:
            for rxIndex in range(0,8):
                rxByte = Dataport.read(1)
                rxByte = np.frombuffer(rxByte, dtype = 'uint8')
                #if the byte received is not in sync with the magicBytes sequence then break and start again
                if rxByte != magicBytes[rxIndex]:
                    break

            if rxIndex == 7: #got all the magicBytes
                lostSync = False
                #read the header frame
                rawRecieveHeaderWithoutMagicBytes = Dataport.read(frameHeaderLength-len(magicBytes))
                rawRecieveHeaderWithoutMagicBytes = np.frombuffer(rawRecieveHeaderWithoutMagicBytes, dtype = 'uint8')
                #concatenate the magic bytes onto the header without magic bytes
                recieveHeader = np.concatenate([magicBytes,rawRecieveHeaderWithoutMagicBytes], axis=0)
                gotHeader = True
                print('BACK IN SYNC')






main()



