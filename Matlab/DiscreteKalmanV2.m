clc;
clear;
load('2peoplecentroidData.mat')

delT = 50*10^-3; % Frame rate
A = [ 1 delT 0 0 ; 0 1 0 0 ; 0 0 1 delT ; 0 0 0 1]; %System Matrix
F = expm(A*delT); %State transition matrix
H = [ 1 0 0 0 ; 0 0 1 0]; %Output Matrix
x = [ 0 ; 0 ; 0 ; 0]; %initial states
centroid(1).P = eye(4);
Q = 0.9*eye(4);
R = [ 1 ; 1 ];
centroid(1).x(:,1) = [0;0;0;0];
centroid(1).xTrack(:,1) = [0;0;0;0];
centroid(1).xpedTrack(:,1) = [0;0;0;0];%initialise the x final in the current frame to zero



rawxycentroidData = load('2peoplecentroidData.mat');
rawxycentroidData= rawxycentroidData.data;


%Tranposed X,Y for better structure
xytransposecentroidData = cellfun(@transpose,rawxycentroidData,'UniformOutput',false); 


rthetacentroidData = xytransposecentroidData;
%Convert from x,y to polar co-ordinates
for i=1:size(xytransposecentroidData,2)
    for j = 1:size(xytransposecentroidData{1,i},2)
        [rthetacentroidData{1,i}(2,j), rthetacentroidData{1,i}(1,j)] = cart2pol(xytransposecentroidData{1,i}(1,j),xytransposecentroidData{1,i}(2,j)); %Converting to polar
    end
end


for i=1:size(rthetacentroidData,2)
    
    if(~(isempty(rthetacentroidData{1,i})))
           
       centroid = data_associate(centroid, rthetacentroidData{1,i}); %new Data frame centroids associated to existing centroids

       for centroidIterator = 1:length(centroid)%add new columns to centroid
           if(isempty(centroid(centroidIterator).x))
               centroid(centroidIterator).x = [0;0;0;0];%initialise the x final in the current frame to zero
               centroid(centroidIterator).P = eye(4);
               centroid(centroidIterator).xpedTrack = [0;0;0;0];%initialise the x final in the current frame to zero
               centroid(centroidIterator).xTrack = [0;0;0;0];%initialise the x final in the current frame to zero
           end
       end
        
       for j =1:length(centroid)
           if(~isempty(centroid(j).currentFrame))
               [centroid(j).xpred, centroid(j).Ppred] = predict(centroid(j).x, centroid(j).P, A, Q);%state prediction
               [centroid(j).nu, centroid(j).S] = innovation(centroid(j).xpred, centroid(j).Ppred, centroid(j).currentFrame , H, R); %innovation function
               [centroid(j).x, centroid(j).P] = innovation_update(centroid(j).xpred, centroid(j).Ppred, centroid(j).nu, centroid(j).S, H);
               centroid(j).xpedTrack(:,end+1) = centroid(j).xpred;
               centroid(j).xTrack(:,end+1) = centroid(j).x;
           else
               [centroid(j).x, centroid(j).Ppred] = predict(centroid(j).x, centroid(j).P, A, Q);%state prediction (ignoring P for now)
               centroid(j).xTrack(:,end+1) = centroid(j).x;
           end
       end
    else
        for j=1:length(centroid)
            [centroid(j).x, centroid(j).Ppred] = predict(centroid(j).x, centroid(j).P, A, Q);%state prediction (ignoring P for now)
            centroid(j).xTrack(:,end+1) = centroid(j).x;
        end
    end       
end

function [xpred, Ppred] = predict(x, P, A, Q)%prediction
xpred = A*x;
Ppred = A*P*A' + Q;
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

function [centroid] = data_associate(centroid, rthetacentroid) %nameFunction better

rthetacentroidCurrent =rthetacentroid %Extract current iteraton from cell array
%%

%% Main Code
for i = 1:length(centroid)
    r1 = (centroid(i).x(1,1)) %Accesses centroids latest updated radius value
    r2 = rthetacentroidCurrent(1,:) %Accesses all the measured data's radius
    theta1 = (centroid(i).x(3,1))
    theta2 = rthetacentroidCurrent(2,:)
    minDist(i,:) = sqrt((r1).^2 + (r2).^2 - 2.*(r1).*(r2).*cos((theta2) - (theta1))) %calculates minimum ditance between two polar
end

[A,orgIndex] = sort(minDist,2) %Ranks each centroids (minDist row) ditance from smallest to highest
[~, rankIndex] = sort(A,1) %Ranks each centroids (minDist column) w.r.t each other
rankIterator = (rankIndex(:,1))' %Transforms to row vector

for i = 1:min(length(rthetacentroidCurrent),length(centroid))
    [~,idx] = min(minDist(rankIterator(i),:));
    centroid(rankIterator(i)).currentFrame = rthetacentroidCurrent(:,idx);
    minDist(:,idx) = []; %remove to find local minimum
    rthetacentroidCurrent(:,idx) = []; %removes from remaining, so that remaining can be added
end
 index = 1;
  if(~isempty(rthetacentroidCurrent)) %If unassociated measured data left, associates it
      for i = ((length(centroid)+1): length(rthetacentroid))
          centroid(i).currentFrame = rthetacentroidCurrent(:,index);
          index = index +1;
      end
  end
end

%Graphing code
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
%     
%     for i=1:size(rthetacentroidData,2)
%         if(~(isempty(rthetacentroidData{1,i})))
%             polarscatter(rthetacentroidData{1,i}(2,1),rthetacentroidData{1,i}(1,1));
%             hold on; %Held so we can compare trails (look for better approach though)
%             title('Before Kalman');
%             rlim([0 6]);
%           %  pause(0); %supposed to simulate time between frames, but hold
%             %on is computationally expensive, so pause (0) is usually used for pace
%         end
%     end
% %     
%     Polar plot of centroi
%     
%     for k = 1 : length(centroid(1).xTrack)
%     %figure(2)
%         polarscatter(centroid(1).xTrack(3,k),centroid(1).xTrack(1,k));
%       hold on;
%         title('After kalman 2')
%         rlim([0 6])
%       %pause(0); %supposed to simulate time between frames, but hold on is computationally expensive, so pause (0) is usually used
%     end
% 