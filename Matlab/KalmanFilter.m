%Kalman Filter Design Preliminary
%Author: Hasnain Cheena 

%Test Dataset
rangeMeasurements = 0:0.1:1;
dopplerMeasurements = 0.1:0.05:0.6;
azimuthMeasurements = 0.1:0.05:0.6;
%each column is a different measurement each each row is a time instance
u = [rangeMeasurements',dopplerMeasurements',azimuthMeasurements'];

deltaT = 0.001;
Ts = deltaT;

%state space model for plant
A = [0 1 0 0;
    2/(deltaT)^2 -2/deltaT 0 0;
    0 0 0 1;
    0 0 2/(deltaT)^2 -2/deltaT];

B = [0 0;
    -2/(deltaT)^2 0;
    0 0;
    0 -2/(deltaT)^2];

C = [1 1 1 0];

D = 0;

%plant model
plant = ss(A,B,C,D,Ts,'inputname',{'u' 'w'},'outputname','y');
%process and measurement covariances
Q = 1;
R = 1;

%observability
observabilityMatrix = obsv(A,C);
observabilityCheck = det(observabilityMatrix);
    
%discrete kalman filter
[kest, L, P, M] = kalman(plant, Q,R);
%kest is the state space model of the kalman estimator
outputEstimate = kest(1,:);

%modified model from matlab website
% a = A;
% b = [B B 0*B];
% c = [C;C];
% d = [0 0 0;0 0 0;0 0 1];
% P = ss(a,b,c,d,-1,'inputname',{'u' 'w' 'v'},'outputname',{'y' 'yv'});
% 
% sys = parallel(P,kalmf,1,1,[],[]);
% 
% SimModel = feedback(sys,1,4,2,1);   % Close loop around input #4 and output #2
% SimModel = SimModel([1 3],[1 2 3]); % Delete yv from I/O list
% 
% w = sqrt(Q)*randn(n,1);
% v = sqrt(R)*randn(n,1);
    
