import serial
import time
import numpy as np
import pandas as pd
import copy
from scipy.io import loadmat

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
        return
    
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
    
    return pointCloud

def offlineParsing(tlvData):

    tlvHeaderLengthInBytes = 8
    pointLengthInBytes = 16
    frameNumber = 0

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
                    positions = pd.DataFrame({'FrameNumber':np.repeat(frameNumber,len(posX)),'X':posX, 'Y':posY})
                    frameNumber += 1
                    positions.to_csv('PointCloudData.csv', index=False,mode='a')
                    