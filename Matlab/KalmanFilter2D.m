clc;
clear;
load('C:\Users\Abin\Documents\Github\OccupancyDetection\Data\centroidData.mat');

delT = 50*10^-3; % Frame rate
A = [ 1 delT 0 0 ; 0 1 0 0 ; 0 0 1 delT ; 0 0 0 1]; %System Matrix
F = expm(A*delT); %State transition matrix
H = [ 1 0 0 0 ; 0 0 1 0]; %Output Matrix
x = [ 0 ; 0 ; 0 ; 0]; %initial states
P = eye(4);
% Qc = eye(4);
% syms T;
% Qtemp = int(F*Qc*F, 0, T);
% Q = subs(Qtemp, T, 50*10^-3);
Q = eye(4); %previous Q doesnt seem to work
R = [ 1 ; 1 ];

rawxycentroidData = load('C:\Users\Abin\Documents\Github\OccupancyDetection\Data\centroidData.mat');
rawxycentroidData= rawxycentroidData.data;
%Accessing cell arrays:  data{1,13}(1,2)

xytransposecentroidData = cellfun(@transpose,rawxycentroidData,'UniformOutput',false); %Tranposed X,Y for better structure
rthetacentroidData = xytransposecentroidData;
doublecount = 1 ; %iterator to find centroid doubles

for i=1:size(xytransposecentroidData,2)
    if(size(xytransposecentroidData{1,i},2)>1)
        xdoublestore(doublecount) = i; %Looking for double ups
        doublecount = doublecount + 1;
    end
    for j = 1:size(xytransposecentroidData{1,i},2)
        [rthetacentroidData{1,i}(2,j), rthetacentroidData{1,i}(1,j)] = cart2pol(xytransposecentroidData{1,i}(1,j),xytransposecentroidData{1,i}(2,j)); %Converting to polar
    end
end

initialFound = [2.8680232;0.78993690]; %Hardcoded for removal of random centroid 
% need more flexible approach for n centroids

%removes 2nd random centroids using minimm distance
%NOTE: this is 'harcoded' for this dataset ; needs flexbile approach for n
%centroids
for i=14:size(rthetacentroidData,2)
    if(size(xytransposecentroidData{1,i},2)>1)
        [~,idx]= min(sum((rthetacentroidData{1,i}-initialFound).^2));
        rthetacentroidData{1,i} = (rthetacentroidData{1,i}(:,idx));
        smallestCentroid = rthetacentroidData{1,i};
    end
end

%main code
for i=1:size(rthetacentroidData,2)
    
    if(~(isempty(rthetacentroidData{1,i})))
        [xpred, Ppred] = predict(x, P, F, Q);%state prediction
        z(1,1)= rthetacentroidData{1,i}(1,1); %hardcoded for 1 centroid
        z(2,1)= rthetacentroidData{1,i}(2,1);
        [nu, S] = innovation(xpred, Ppred, z , H, R); %innovation function
        [x, P] = innovation_update(xpred, Ppred, nu, S, H);%update function
        xtracked(:,i) = x; %stores each step
    end
    
    %Plots centroids on xy coordiates
    % for i=1:size(xytransposecentroidData,2)
    %     if(~(isempty(xytransposecentroidData{1,i})))
    %         xytransposecentroidData{1,i}(1,1);
    %         xytransposecentroidData{1,i}(2,1);
    %         scatter(xytransposecentroidData{1,i}(1,1),xytransposecentroidData{1,i}(2,1));
    %         axis([-6 6 0 6])
    %         pause(50*10^-3);
    %         refreshdata;
    %     end
    % end
    
    % plot centroid polar plot before Kalman
    
%     for i=1:size(rthetacentroidData,2)
%         if(~(isempty(rthetacentroidData{1,i})))
%             polarscatter(rthetacentroidData{1,i}(2,1),rthetacentroidData{1,i}(1,1));
%             hold on; %Held so we can compare trails (look for better approach though)
%             title('Before Kalman');
%             rlim([0 6]);
%             %pause(50*10^-3); %supposed to simulate time between frames, but hold
%             %on is computationally expensive, so pause (0) is usually used for pace
%         end
%     end
%     
    %Polar plot
    
%     for k = 1 : length(xtracked)
%         figure(2)
%         polarscatter(xtracked(3,k),xtracked(1,k));
%         hold on;
%         title('After kalman 2')
%         rlim([0 6])
%         %pause(50*10^-3); %supposed to simulate time between frames, but hold
%         %on is computationally expensive, so pause (0) is usually used
%     end
    
    
end

function [xpred, Ppred] = predict(x, P, F, Q)%prediction
xpred = F*x;
Ppred = F*P*F' + Q;
end

function [nu, S] = innovation(xpred, Ppred, z, H, R)
nu = z - H*xpred; %% innovation
S = R + H*Ppred*H'; %% innovation covariance
end

function [xnew, Pnew] = innovation_update(xpred, Ppred, nu, S, H)
K = Ppred*H'*inv(S); %% Kalman gain
xnew = xpred + K*nu; %% new state
Pnew = Ppred - K*S*K'; %% new covariance

end
