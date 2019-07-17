import numpy as np
import copy
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt

rawxycentroidData = (loadmat('C:\\Users\\Abin\\Desktop\\Part4\\Research\\Development\\Kalman\\2peoplecentroidData.mat'))['data'][0]

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

#associate function
def data_associate(centroidPred, rthetacentroid):
    rthetacentroidCurrent = rthetacentroid
    centpredCol = np.size(centroidPred,1)
    rthetaCol = np.size(rthetacentroid,1)

    for i in list(range(0,centpredCol)):
        r1 = centroidPred[0][i]
        r2 = rthetacentroid[0]
        theta1 = centroidPred[2][i] #HARDCODE: theta is 2nd index
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


#initialise matrices 
centroidX =np.zeros((4,1))
centroidP = []
delT = 0.0500
A = np.array([[1,delT,0,0], [0,1,0,0], [0,0,1,delT], [0,0,0,1]])
H = np.array([[1,0,0,0],[0,0,1,0]])
P = np.identity(4);
Q = np.multiply(0.9,np.identity(4))
R = np.array([[1],[1]])
centroidP.extend([P])

plt.figure()
theta1 = []
radius1 = []
theta2 = []
radius2 = []
ax = plt.subplot(111, projection='polar')

for currentrawxycentroidData in rawxycentroidData:
    
    xytransposecentroidData = np.transpose(currentrawxycentroidData) #Tranposed
    rthetacentroidData=xytransposecentroidData
    if (xytransposecentroidData.size != 0): 
        [rthetacentroidData[0,:],rthetacentroidData[1,:]] = cart2pol(xytransposecentroidData[0,:],xytransposecentroidData[1,:])
    if((rthetacentroidData.size != 0)):
        currentFrame = data_associate(centroidX, rthetacentroidData)
        addittionalCentroids = (np.size(rthetacentroidData,1)-np.size(centroidX,1))
        if(addittionalCentroids>0):
            centroidX = np.pad(centroidX, ((0,0),(0,addittionalCentroids)), 'constant') #initialises previous iteration to zer
            for newFrameIndex in list((range(0, addittionalCentroids))):
                centroidP.extend([P])
                print(type(centroidP))
        for currentFrameIndex in list((range(0,np.size(currentFrame,1)))):
            if(not(np.isnan(currentFrame[0,currentFrameIndex]))):
                [xpred, Ppred] = predict(centroidX[:,currentFrameIndex], centroidP[currentFrameIndex], A, Q)
                [nu, S] = innovation(xpred, Ppred, currentFrame[:, currentFrameIndex], H, R)
                [centroidX[:,currentFrameIndex],  centroidP[currentFrameIndex]] = innovation_update(xpred, Ppred, nu, S, H)
             #   print("for currentFrame", currentFrame[:, currentFrameIndex], "Ppred is",Ppred, "S is", S, "the update is",centroidX[:,currentFrameIndex])
            else:
                [centroidX[:,currentFrameIndex], Ppred] = predict(centroidX[:,currentFrameIndex], P, A, Q)
    else:
        for noFrameIndex in list((range(0,np.size(centroidX,1)))):
            [centroidX[:,noFrameIndex], Ppred] = predict(centroidX[:,noFrameIndex], P, A, Q)

    if(np.size(centroidX,1) > 1):
        theta2.append(centroidX[2][1])
        radius2.append(centroidX[0][1])
    theta1.append(centroidX[2][0])
    radius1.append(centroidX[0][0])

   # print(centroidX)

##person1 = np.vstack((radius1,theta1))
person2 = np.vstack((radius2,theta2))
##print(person1)
##df = pd.DataFrame(person1)
df = pd.DataFrame(person2)
##df.to_csv('person1.csv',index=False)
df.to_csv('person2.csv',index=False)
ax.scatter(theta2, radius2)
plt.figure()
bx = plt.subplot(111, projection='polar')
bx.scatter(theta1, radius1)
plt.show()
