clc;
clear;
load('C:\Users\Abin\Documents\Github\OccupancyDetection\Data\centroidData.mat');

%   for k = 1 : length(xtracked)
% polarscatter(xtracked(2,k),xtracked(1,k));
% pause( 0.005 );
% end
% hold off

%Assumptions
% The only difference is the way in which the measurements are fed since
% its fed element by element
%%% Matlab script to simulate data and process usiung Kalman filter

delT = 50*10^-3; % Frame rate
A = [ 1 delT 0 0 ; 0 1 0 0 ; 0 0 1 delT ; 0 0 0 1]; %System Matrix
F = expm(A*delT); %State transition matrix
H = [ 1 0 0 0 ; 0 0 1 0]; %Output Matrix
x = [ 0 ; 0 ; 0 ; 0]; %initial states
P = eye(4);
Qc = eye(4);
syms T;
Qtemp = int(F*Qc*F, 0, T);
Q = subs(Qtemp, T, 50*10^-3);
Q = eye(4); %previous Q doesnt seem to work
R = [ 1 ; 1 ];

rawxycentroidData = load('C:\Users\Abin\Documents\Github\OccupancyDetection\Data\centroidData.mat');
rawxycentroidData= rawxycentroidData.data;
%Accessing cell arrays:  data{1,13}(1,2)

xytransposecentroidData = cellfun(@transpose,rawxycentroidData,'UniformOutput',false);
rthetacentroidData = xytransposecentroidData;


for i=1:size(xytransposecentroidData,2)
   for j = 1:size(xytransposecentroidData{1,i},2)
       [rthetacentroidData{1,i}(2,j), rthetacentroidData{1,i}(1,j)] = cart2pol(xytransposecentroidData{1,i}(1,j),xytransposecentroidData{1,i}(2,j));
   end
end

% for i=1:size(xytransposecentroidData,2)
%     if(~(isempty(xytransposecentroidData{1,i})))
%         xytransposecentroidData{1,i}(1,1);
%         xytransposecentroidData{1,i}(2,1);
%         axis([-6 6 0 6])
%         hold all;
%         scatter(xytransposecentroidData{1,i}(1,1),xytransposecentroidData{1,i}(2,1));
%         pause(50*10^-3);
%     end
% end

% plot centroid polar plot
% for i=1:size(rthetacentroidData,2)
%     if(~(isempty(rthetacentroidData{1,i})))
%         polarscatter(rthetacentroidData{1,i}(2,1),rthetacentroidData{1,i}(1,1));
%         pause(50*10^-3);
%     end
% end



for i=1:size(rthetacentroidData,2)
    if(~(isempty(rthetacentroidData{1,i})))
        [xpred, Ppred] = predict(x, P, F, Q);
        z(1,1)= rthetacentroidData{1,i}(1,1);
        z(2,1)= rthetacentroidData{1,i}(2,1);
        [nu, S] = innovation(xpred, Ppred, z , H, R);
        [x, P] = innovation_update(xpred, Ppred, nu, S, H);
        xtracked(:,i) = x;
    end
end

function [xpred, Ppred] = predict(x, P, F, Q)
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
