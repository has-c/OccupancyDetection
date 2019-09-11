clear, clc

%% SETUP SELECTIONS

% 1. sceneRun needs to be specied.
% Default is to use GUI to setup scene
% To programmatically setup the scene follow the example in Prgm_2box.

% Scene Run choice: {'GUI_Setup' | 'Prgm_2box' | 'Prgm_MaxFOV','My_Scene'};
sceneRun = 'NotGUI_Setup';
% loadCfg = 1;
%
% [strPorts numPorts] = get_com_ports();

if (~strcmp(sceneRun,'GUI_Setup'))
    %%%%% EDIT COM PORTS %%%%%%
    controlSerialPort = 4;
    dataSerialPort = 3;
    loadCfg = 1;
end

fig1 = figure();

tlvStream = {};

%% Serial setup
if (~strcmp(sceneRun,'GUI_Setup'))
    %Configure data UART port with input buffer to hold 100+ frames
    hDataSerialPort = configureDataSport(dataSerialPort, 65536);
    %configure control port
    hControlSerialPort = configureControlPort(controlSerialPort);
    disp('COM STATUS: Ports connected');
    
    %Read Chirp Configuration file
    configurationFileName = 'mmw_pplcount_demo_default.cfg';
    cliCfg = readCfg(configurationFileName);
    Params = parseCfg(cliCfg);
    
    %Send Configuration Parameters to IWR16xx
    if(loadCfg)
        mmwDemoCliPrompt = char('mmwDemo:/>');
        
        %Send CLI configuration to IWR16xx
        fprintf('Sending configuration from %s file to IWR16xx ...\n', configurationFileName);
        for k=1:length(cliCfg)
            fprintf(hControlSerialPort, cliCfg{k});
            fprintf('%s\n', cliCfg{k});
            echo = fgetl(hControlSerialPort); % Get an echo of a command
            done = fgetl(hControlSerialPort); % Get "Done"
            disp(done);
            prompt = fread(hControlSerialPort, size(mmwDemoCliPrompt,2)); % Get the prompt back
        end
        fclose(hControlSerialPort);
        delete(hControlSerialPort);
        clear hControlSerialPort;
    end
end

%% Init variables
trackerRun = 'Target';
colors='brgcm';
labelTrack = 0;

%sensor parameters
sensor.rangeMax = Params.dataPath.numRangeBins*Params.dataPath.rangeIdxToMeters; %Params come from setup.m
sensor.rangeMin = 1;
sensor.azimuthFoV = 120*pi/180; %120 degree FOV in horizontal direction
sensor.framePeriod = Params.frameCfg.framePeriodicity;
sensor.maxURadialVelocity = 20;
sensor.angles = linspace(-sensor.azimuthFoV/2, sensor.azimuthFoV/2, 128);

peopleCountTotal = 0;

rxData = zeros(10000,1,'uint8');

maxNumTracks = 20;
maxNumPoints = 250;

%% Data structures
syncPatternUINT64 = typecast(uint16([hex2dec('0102'),hex2dec('0304'),hex2dec('0506'),hex2dec('0708')]),'uint64');
syncPatternUINT8 = typecast(uint16([hex2dec('0102'),hex2dec('0304'),hex2dec('0506'),hex2dec('0708')]),'uint8'); %Frameheader

frameHeaderStructType = struct(... %Frame header contains information about the data specs of the packet
    'sync',             {'uint64', 8}, ... % See syncPatternUINT64 below
    'version',          {'uint32', 4}, ...
    'platform',         {'uint32', 4}, ...
    'timestamp',        {'uint32', 4}, ... % 600MHz clocks
    'packetLength',     {'uint32', 4}, ... % In bytes, including header
    'frameNumber',      {'uint32', 4}, ... % Starting from 1
    'subframeNumber',   {'uint32', 4}, ...
    'chirpMargin',      {'uint32', 4}, ... % Chirp Processing margin, in ms
    'frameMargin',      {'uint32', 4}, ... % Frame Processing margin, in ms
    'uartSentTime' ,    {'uint32', 4}, ... % Time spent to send data, in ms
    'trackProcessTime', {'uint32', 4}, ... % Tracking Processing time, in ms
    'numTLVs' ,         {'uint16', 2}, ... % Number of TLVs in this frame
    'checksum',         {'uint16', 2});    % Header checksum

tlvHeaderStruct = struct(...
    'type',             {'uint32', 4}, ... % TLV object Type
    'length',           {'uint32', 4});    % TLV object Length, in bytes, including TLV header

% Point Cloud TLV object consists of an array of points.
% Each point has a structure defined below
pointStruct = struct(...
    'range',            {'float', 4}, ... % Range, in m
    'angle',            {'float', 4}, ... % Angel, in rad
    'doppler',          {'float', 4}, ... % Doplper, in m/s
    'snr',              {'float', 4});    % SNR, ratio
% Target List TLV object consists of an array of targets.
% Each target has a structure define below
targetStruct = struct(...
    'tid',              {'uint32', 4}, ... % Track ID
    'posX',             {'float', 4}, ... % Target position in X dimension, m
    'posY',             {'float', 4}, ... % Target position in Y dimension, m
    'velX',             {'float', 4}, ... % Target velocity in X dimension, m/s
    'velY',             {'float', 4}, ... % Target velocity in Y dimension, m/s
    'accX',             {'float', 4}, ... % Target acceleration in X dimension, m/s2
    'accY',             {'float', 4}, ... % Target acceleration in Y dimension, m/s
    'EC',               {'float', 9*4}, ... % Tracking error covariance matrix, [3x3], in range/angle/doppler coordinates
    'G',                {'float', 4});    % Gating function gain

frameHeaderLengthInBytes = lengthFromStruct(frameHeaderStructType); %for finding frame header in datapacket
tlvHeaderLengthInBytes = lengthFromStruct(tlvHeaderStruct); %for finding TLV header in data packet
pointLengthInBytes = lengthFromStruct(pointStruct);%for finding point in data packet
targetLengthInBytes = lengthFromStruct(targetStruct); %for finding taget in data packet
indexLengthInBytes = 1; %for finding target indexes

lostSync = 0;
gotHeader = 0;
outOfSyncBytes = 0;
runningSlow = 0;
maxBytesAvailable = 0;
point3D = [];

frameStatStruct = struct('targetFrameNum', [], 'bytes', [], 'numInputPoints', 0, 'numOutputPoints', 0, 'timestamp', 0, 'start', 0, 'benchmarks', [], 'done', 0, ...
    'pointCloud', [], 'targetList', [], 'indexArray', []); %FRAME PACKET STRUCTURE
fHist = repmat(frameStatStruct, 1, 10000); %INITIALIZES DATA FRAME ARRAY


optimize = 1;
skipProcessing = 0;
frameNum = 1;
frameNumLogged = 1;
fprintf('------------------\n');

update = 0;
%% Main
while(isvalid(hDataSerialPort))
    
    
    while(lostSync == 0 && isvalid(hDataSerialPort)) %if in sync and there is data in the port object
        
        %% checks if we have a valid frame header
        frameStart = tic; %framestart= start of stopwatch
        fHist(frameNum).timestamp = frameStart; %stores the start time in the frame packet data variable (fhist) in the franeNum elemenet under the attribute of time stamp
        
        bytesAvailable = get(hDataSerialPort,'BytesAvailable'); %Extracts how many bytes are in the serial port object 'hDataserialport"
        if(bytesAvailable > maxBytesAvailable) % stores the max the highest bytes available @@@@
            maxBytesAvailable = bytesAvailable;
        end
        fHist(frameNum).bytesAvailable = bytesAvailable; %stores the number of bytes availabe in the frame packet data variable (fhist) in the franeNum elemenet under the attribute of time stamp
        if(gotHeader == 0) %ensures that the frame header is read first
            %Read the header first
            [rxHeader, byteCount] = fread(hDataSerialPort, frameHeaderLengthInBytes, 'uint8'); %fread reads the hdatasearialport object for the frameheader bytes into an array rxHeader
%             rawframeHeader{end+1} = rxHeader;
        end
        
        fHist(frameNum).start = 1000*toc(frameStart); % toc -stops and outputs delayed time; puts time in millseconds
        
        magicBytes = typecast(uint8(rxHeader(1:8)), 'uint64'); %Stores the first 8 numbers or the right patern
        if(magicBytes ~= syncPatternUINT64) %Ensures that it matches (esp for first iteration)
            reason = 'No SYNC pattern';
            lostSync = 1; %Is lost and starts looking
            break;
        end
        if(byteCount ~= frameHeaderLengthInBytes) %ensures header is the right size @@@@@
            reason = 'Header Size is wrong';
            lostSync = 1; %If the header is not the right size, start finding right pattern/magic number again ie ur lost
            break;
        end
        if(validateChecksum(rxHeader) ~= 0) %Error checking
            reason = 'Header Checksum is wrong';
            lostSync = 1; %start looking again
            break;
        end
        
        frameHeader = readToStruct(frameHeaderStructType, rxHeader); %reads the raw rx header data into the frame header using the frameHeaderStructType properties
        
        %getting in sync again
        if(gotHeader == 1)
            if(frameHeader.frameNumber > targetFrameNum)
                targetFrameNum = frameHeader.frameNumber;
                disp(['Found sync at frame ',num2str(targetFrameNum),'(',num2str(frameNum),'), after ', num2str(1000*toc(lostSyncTime),3), 'ms']);
                gotHeader = 0;
            else %lost sync and start again
                reason = 'Old Frame';
                gotHeader = 0;
                lostSync = 1;
                break;
            end
        end
        
        %% We have a valid header
        targetFrameNum = frameHeader.frameNumber; %sets the frame number that we are currenty at
        fHist(frameNum).targetFrameNum = targetFrameNum; %sets the frame number within the larger fHist structure
        fHist(frameNum).header = frameHeader; %assigns the frame header to the larger fHist structure
        
        dataLength = frameHeader.packetLength - frameHeaderLengthInBytes; %total packet length subtracted from frame header length to get the length of the actual useful data
        
        fHist(frameNum).bytes = dataLength; %save this data length within the larger fHist structure
        numInputPoints = 0;
        numTargets = 0;
        mIndex = [];
        
        if(dataLength > 0) %If there is valid data in the packet
            %Read all packet
            
            [rxData, byteCount] = fread(hDataSerialPort, double(dataLength), 'uint8'); %read the rest of the packet
            tlvStream{end+1} = rxData;
            if(byteCount ~= double(dataLength)) %if the number of bytes read from above is not equal to the preset data length then something is wrong
                reason = 'Data Size is wrong';
                lostSync = 1;
                break;
            end
            offset = 0;%reset for the next iteration
            
            fHist(frameNum).benchmarks(1) = 1000*toc(frameStart); %timing benchmark
            
            %% TLV Parsing
            %each TLV has a fixed header (8 bytes). the header contains the
            %type and length information
            for nTlv = 1:frameHeader.numTLVs %loop through the number of TLV's within this frame
                %TLV header parsing
                tlvType = typecast(uint8(rxData(offset+1:offset+4)), 'uint32'); %what TLV is being parsed; TLV Types = Point Cloud TLV, Target List TLV, Target Index TLV
                tlvLength = typecast(uint8(rxData(offset+5:offset+8)), 'uint32'); %how long is the TLV being parsed
                %if the length of the TLV header is wrong then we are lost
                if(tlvLength + offset > dataLength)
                    reason = 'TLV Size is wrong';
                    lostSync = 1;
                    break;
                end
                offset = offset + tlvHeaderLengthInBytes; %change offset to now point to the beginning of the TLV value data
                valueLength = tlvLength - tlvHeaderLengthInBytes; %calculate length of the actual TLV data
                switch(tlvType) %depending on TLV Type run a particular portion of code
                    
                    case 6  % Point Cloud TLV
                        numInputPoints = valueLength/pointLengthInBytes; %calculate number of avaliable points => total TLV frame length / length of one piece of point data
                        if(numInputPoints > 0) %actually have some points to parse
                            % Get Point Cloud from the sensor
                            p = typecast(uint8(rxData(offset+1: offset+valueLength)),'single'); %get all avaliable point cloud data from the sensor
%                             rawPData{end+1} = p;
                            pointCloud = reshape(p,4, numInputPoints); %form point cloud, resultant matrix is 4 x numInputPoints in size
                           
%                             pointCloudData{end+1} = pointCloud;
                            %row 1 = raw magnitude data (range data)
                            %row 2 = raw angle data (azimuth data)
                            %row 3 = raw doppler data
                            %row 4 = raw snr data
                            
%                             staticInd = (pointCloud(3,:) == 0);   
%                             clutterPoints = pointCloud(1:2,staticInd);
%                             clutterInd = ismember(pointCloud(1:2,:)', clutterPoints', 'rows');
%                             clutterInd = clutterInd' & staticInd;
%                             pointCloud = pointCloud(1:3,~clutterInd);
                            
                            %posAll 1st row = Rsin(theta)
                            %posAll 2nd row = Rcos(theta)
                            posAll = [pointCloud(1,:).*sin(pointCloud(2,:)); pointCloud(1,:).*cos(pointCloud(2,:))]; %calculate y(row 1) x(row 2) positions => resultant matrix is 2 by numInputPoints in size
%                             snrAll = pointCloud(4,:); %extract the signal to noise ratio from the point cloud
%                             posAllData{end+1} = posAll;

                            
                            % Remove out of Range, Behind the Walls, out of field of view (FOV) points
                            %find index of point cloud that is within the
                            %effective range of the sensor
                            inRangeInd = (pointCloud(1,:) > 1) & (pointCloud(1,:) < 6) & ... %max and min effective range of the sensor
                                (pointCloud(2,:) > -50*pi/180) &  (pointCloud(2,:) < 50*pi/180); %max and min angle of the sensor
                            
                            pointCloudInRange = pointCloud(:,inRangeInd); %extract portion of the point cloud that is within the sensor limits
                            posInRange = posAll(:,inRangeInd); %extract positions (x,y) that are within the sensor limits
%                             posData{end+1} = posInRange;

                            numOutputPoints = size(pointCloud,2); % output number of coloumns in the point cloud
                        end
                        offset = offset + valueLength; %updates offset
                        
                    case 7 % Target List TLV
                        
                        numTargets = valueLength/targetLengthInBytes;  %number of actual targets => total TLV data length / data length of one target
                        TID = zeros(1,numTargets); %target ID matrix
                        S = zeros(6, numTargets); %contains posX(m), posY(m), velX(ms^-1), velY(ms^-1), accX(ms^-2) and accY(ms^-2) of target
                        EC = zeros(9, numTargets); % error covariance matrix
                        G = zeros(1,numTargets); % gating gain
                        for n=1:numTargets
                            TID(n)  = typecast(uint8(rxData(offset+1:offset+4)),'uint32');      %1x4=4bytes, extract target ID from data
                            S(:,n)  = typecast(uint8(rxData(offset+5:offset+28)),'single');     %6x4=24bytes, extracts kinematic information
                            EC(:,n) = typecast(uint8(rxData(offset+29:offset+64)),'single');    %9x4=36bytes, extracts the covariance
                            G(n)    = typecast(uint8(rxData(offset+65:offset+68)),'single');    %1x4=4bytes, extracts gating gain
                            offset = offset + 68; %increments offset
                        end
                        
                        
                    case 8
                        % Target Index TLV - list of target ID's for the
                        % previous point cloud (from previous frame)
                        numIndices = valueLength/indexLengthInBytes; %number of indices that relate to targets
                        mIndex = typecast(uint8(rxData(offset+1:offset+numIndices)),'uint8'); %extract target Index data, array of target indices
                        offset = offset + valueLength; %update offset
                end
            end
        end
        %if no actual kinematic data is present reset parameters
        if(numInputPoints == 0)
            numOutputPoints = 0;
            pointCloud = single(zeros(4,0));
            posAll = [];
            posInRange = [];
        end
        %if no target data is present reset parameters
        if(numTargets == 0)
            TID = [];
            S = [];
            EC = [];
            G = [];
        end
        
        %otherwise, assign the various data items into the main dataframe
        fHist(frameNum).numInputPoints = numInputPoints;
        fHist(frameNum).numOutputPoints = numOutputPoints;
        fHist(frameNum).numTargets = numTargets;
        fHist(frameNum).pointCloud = pointCloud;
        fHist(frameNum).targetList.numTargets = numTargets;
        fHist(frameNum).targetList.TID = TID;
        fHist(frameNum).targetList.S = S;
        %if you don't want to optimise to lower the covariance error then
        %assign the raw value to the main dataframe
        if(~optimize)
            fHist(frameNum).targetList.EC = EC;
        end
        fHist(frameNum).targetList.G = G;
        fHist(frameNum).indexArray = mIndex;
        
        % Plot pointCloud
        fHist(frameNum).benchmarks(2) = 1000*toc(frameStart); %bench timing feature
        
        fHist(frameNum).done = 1000*toc(frameStart); %benchmark timing
        
        %time to read the next frame
        frameNum = frameNum + 1;
        frameNumLogged = frameNumLogged + 1;
        if(frameNum > 10000)
            frameNum = 1;
        end
        
        %create a matrix that contains the x,y points and the doppler data
        point3D = [posAll; pointCloud(3,:)];
        
        %check if system is running slowly
        if(bytesAvailable > 32000)
            runningSlow  = 1;
        elseif(bytesAvailable < 1000)
            runningSlow = 0;
        end
        
        if(runningSlow)
            % Don't pause, we are slow
        else
            pause(0.01);
         end
        
       %plot raw point cloud positions and targets
%        if (~isempty(pointCloud) && ~isempty(S))
%            xPos = posAll(1,:); 
%            yPos = posAll(2,:);
%            xTar = S(1,:);
%            yTar = S(2,:);
%            
%            figure(fig1);
%            subplot(2,1,1);
%            scatter(xPos, yPos);
%            xlim([-6 6]);
%            ylim([0 6]);
%            xlabel('x Position');
%            ylabel('y Position'); 
%            
%            subplot(2,1,2);
%            scatter(xTar,yTar,'xr');
%            xlim([-6 6]);
%            ylim([0 6]);
%            xlabel('x Position');
%            ylabel('y Position'); 
%            
%        else
%            xPos = [];
%            yPos = [];
%            xTar = [];
%            yTar = [];
%            
%            figure(fig1);
%            subplot(2,1,1);
%            scatter(xPos, yPos);
%            xlim([-6 6]);
%            ylim([0 6]);
%            xlabel('x Position');
%            ylabel('y Position');
%            
%            subplot(2,1,2);
%            scatter(xTar,yTar,'xr');
%            xlim([-6 6]);
%            ylim([0 6]);
%            xlabel('x Position');
%            ylabel('y Position');
%            
%            
%            
%        end

    end
    
    if(targetFrameNum)
        lostSyncTime = tic;
        bytesAvailable = get(hDataSerialPort,'BytesAvailable');
        disp(['Lost sync at frame ', num2str(targetFrameNum),'(', num2str(frameNum), '), Reason: ', reason, ', ', num2str(bytesAvailable), ' bytes in Rx buffer']);
    else
        errordlg('Port sync error: Please close and restart program');
    end
    %{
    % To catch up, we read and discard all uart data
    bytesAvailable = get(hDataSerialPort,'BytesAvailable');
    disp(bytesAvailable);
    [rxDataDebug, byteCountDebug] = fread(hDataSerialPort, bytesAvailable, 'uint8');
    %}
    while(lostSync) %Waits till it finds the first pattern of numbers and therefore is no longer lost
        for n=1:8 %Checks if the digits read match the frame header
            
            [rxByte, byteCount] = fread(hDataSerialPort, 1, 'uint8');
            if(rxByte ~= syncPatternUINT8(n))
                outOfSyncBytes = outOfSyncBytes + 1; %Keeps track of out of sync @@@@@
                break;
            end
        end
        if(n == 8)
            lostSync = 0; %if it has found the first pattern/magic number, then no longer lost
            frameNum = frameNum + 1; %moves the framepacket indexer to the next (bytes)
            if(frameNum > 10000) %Resets frame index if it has reached the end of the frame data file
                frameNum = 1;
            end
            
            [header, byteCount] = fread(hDataSerialPort, frameHeaderLengthInBytes - 8, 'uint8');
            rxHeader = [syncPatternUINT8'; header];
            byteCount = byteCount + 8;
            gotHeader = 1;
        end
    end
    
end


%% Helper functions

function [strPorts numPorts] = get_com_ports()

command = 'wmic path win32_pnpentity get caption /format:list | find "COM"';
[status, cmdout] = system (command);
UART_COM = regexp(cmdout, 'UART\s+\(COM[0-9]+', 'match');
UART_COM = (regexp(UART_COM, 'COM[0-9]+', 'match'));
DATA_COM = regexp(cmdout, 'Data\s+Port\s+\(COM[0-9]+', 'match');
DATA_COM = (regexp(DATA_COM, 'COM[0-9]+', 'match'));

n = length(UART_COM);
if (n==0)
    errordlg('Error: No Device Detected')
    return
else
    CLI_PORT = zeros(n,1);
    S_PORT = zeros(n,1);
    strPorts = {};
    for i=1:n
        temp = cell2mat(UART_COM{1,i});
        strPorts{i,1} = temp;
        CLI_PORT(i,1) = str2num(temp(4:end));
        temp = cell2mat(DATA_COM{1,i});
        strPorts{i,2} = temp;
        S_PORT(i,1) = str2num(temp(4:end));
    end
    
    CLI_PORT = sort(CLI_PORT);
    S_PORT = sort(S_PORT);
    numPorts = [CLI_PORT, S_PORT];
end
end

%Display Chirp parameters in table on screen
function h = displayChirpParams(Params, Position, hFig)

dat =  {'Start Frequency (Ghz)', Params.profileCfg.startFreq;...
    'Slope (MHz/us)', Params.profileCfg.freqSlopeConst;...
    'Samples per chirp', Params.profileCfg.numAdcSamples;...
    'Chirps per frame',  Params.dataPath.numChirpsPerFrame;...
    'Frame duration (ms)',  Params.frameCfg.framePeriodicity;...
    'Sampling rate (Msps)', Params.profileCfg.digOutSampleRate / 1000;...
    'Bandwidth (GHz)', Params.profileCfg.freqSlopeConst * Params.profileCfg.numAdcSamples /...
    Params.profileCfg.digOutSampleRate;...
    'Range resolution (m)', Params.dataPath.rangeResolutionMeters;...
    'Velocity resolution (m/s)', Params.dataPath.dopplerResolutionMps;...
    'Number of Rx (MIMO)', Params.dataPath.numRxAnt; ...
    'Number of Tx (MIMO)', Params.dataPath.numTxAnt;};
columnname =   {'Chirp Parameter (Units)      ', 'Value'};
columnformat = {'char', 'numeric'};

h = uitable('Parent',hFig,'Units','normalized', ...
    'Position', Position, ...
    'Data', dat,...
    'ColumnName', columnname,...
    'ColumnFormat', columnformat,...
    'ColumnWidth', 'auto',...
    'RowName',[]);
end

function [P] = parseCfg(cliCfg)
P=[];
for k=1:length(cliCfg)
    C = strsplit(cliCfg{k});
    if strcmp(C{1},'channelCfg')
        P.channelCfg.txChannelEn = str2double(C{3});
        P.dataPath.numTxAzimAnt = bitand(bitshift(P.channelCfg.txChannelEn,0),1) +...
            bitand(bitshift(P.channelCfg.txChannelEn,-1),1);
        P.dataPath.numTxElevAnt = 0;
        P.channelCfg.rxChannelEn = str2double(C{2});
        P.dataPath.numRxAnt = bitand(bitshift(P.channelCfg.rxChannelEn,0),1) +...
            bitand(bitshift(P.channelCfg.rxChannelEn,-1),1) +...
            bitand(bitshift(P.channelCfg.rxChannelEn,-2),1) +...
            bitand(bitshift(P.channelCfg.rxChannelEn,-3),1);
        P.dataPath.numTxAnt = P.dataPath.numTxElevAnt + P.dataPath.numTxAzimAnt;
        
    elseif strcmp(C{1},'dataFmt')
    elseif strcmp(C{1},'profileCfg')
        P.profileCfg.startFreq = str2double(C{3});
        P.profileCfg.idleTime =  str2double(C{4});
        P.profileCfg.rampEndTime = str2double(C{6});
        P.profileCfg.freqSlopeConst = str2double(C{9});
        P.profileCfg.numAdcSamples = str2double(C{11});
        P.profileCfg.digOutSampleRate = str2double(C{12}); %uints: ksps
    elseif strcmp(C{1},'chirpCfg')
    elseif strcmp(C{1},'frameCfg')
        P.frameCfg.chirpStartIdx = str2double(C{2});
        P.frameCfg.chirpEndIdx = str2double(C{3});
        P.frameCfg.numLoops = str2double(C{4});
        P.frameCfg.numFrames = str2double(C{5});
        P.frameCfg.framePeriodicity = str2double(C{6});
    elseif strcmp(C{1},'guiMonitor')
        P.guiMonitor.detectedObjects = str2double(C{2});
        P.guiMonitor.logMagRange = str2double(C{3});
        P.guiMonitor.rangeAzimuthHeatMap = str2double(C{4});
        P.guiMonitor.rangeDopplerHeatMap = str2double(C{5});
    end
end
P.dataPath.numChirpsPerFrame = (P.frameCfg.chirpEndIdx -...
    P.frameCfg.chirpStartIdx + 1) *...
    P.frameCfg.numLoops;
P.dataPath.numDopplerBins = P.dataPath.numChirpsPerFrame / P.dataPath.numTxAnt;
P.dataPath.numRangeBins = pow2roundup(P.profileCfg.numAdcSamples);
P.dataPath.rangeResolutionMeters = 3e8 * P.profileCfg.digOutSampleRate * 1e3 /...
    (2 * P.profileCfg.freqSlopeConst * 1e12 * P.profileCfg.numAdcSamples);
P.dataPath.rangeIdxToMeters = 3e8 * P.profileCfg.digOutSampleRate * 1e3 /...
    (2 * P.profileCfg.freqSlopeConst * 1e12 * P.dataPath.numRangeBins);
P.dataPath.dopplerResolutionMps = 3e8 / (2*P.profileCfg.startFreq*1e9 *...
    (P.profileCfg.idleTime + P.profileCfg.rampEndTime) *...
    1e-6 * P.dataPath.numDopplerBins * P.dataPath.numTxAnt);
end


function [] = dispError()
disp('Serial Port Error!');
end

function exitPressFcn(hObject, ~)
setappdata(hObject, 'exitKeyPressed', 1);
end

function checkPlotTabs(hObject, eventData, hTabGroup)
if(hObject.Value)
    % get children
    children = hTabGroup(3).Children;
    hTabGroup(3).UserData = children; %save children to restore
    
    % combine tab group
    for t=1:length(children)
        set(children(t),'Parent',hTabGroup(2));
    end
    
    % resize tab group
    hTabGroup(2).UserData = hTabGroup(2).Position; %save position to restore
    hTabGroup(2).Position = [0.2 0 0.8 1];
    hTabGroup(3).Visible = 'off';
else
    % restore children
    children = hTabGroup(3).UserData;
    
    % move tab group
    for t=1:length(children)
        set(children(t),'Parent',hTabGroup(3));
    end
    
    % resize tab group
    hTabGroup(2).Position = hTabGroup(2).UserData;
    hTabGroup(3).Visible = 'on';
end

end

function [sphandle] = configureDataSport(comPortNum, bufferSize)
if ~isempty(instrfind('Type','serial'))
    disp('Serial port(s) already open. Re-initializing...');
    delete(instrfind('Type','serial'));  % delete open serial ports.
end
comPortString = ['COM' num2str(comPortNum)];
sphandle = serial(comPortString,'BaudRate',921600);
set(sphandle,'Terminator', '');
set(sphandle,'InputBufferSize', bufferSize);
set(sphandle,'Timeout',10);
set(sphandle,'ErrorFcn',@dispError);
fopen(sphandle);
end

function [sphandle] = configureControlPort(comPortNum)
%if ~isempty(instrfind('Type','serial'))
%    disp('Serial port(s) already open. Re-initializing...');
%    delete(instrfind('Type','serial'));  % delete open serial ports.
%end
comPortString = ['COM' num2str(comPortNum)];
sphandle = serial(comPortString,'BaudRate',115200);
set(sphandle,'Parity','none')
set(sphandle,'Terminator','LF')
fopen(sphandle);
end

function config = readCfg(filename)
config = cell(1,100);
fid = fopen(filename, 'r');
if fid == -1
    fprintf('File %s not found!\n', filename);
    return;
else
    fprintf('Opening configuration file %s ...\n', filename);
end
tline = fgetl(fid);
k=1;
while ischar(tline)
    config{k} = tline;
    tline = fgetl(fid);
    k = k + 1;
end
config = config(1:k-1);
fclose(fid);
end

function length = lengthFromStruct(S)
fieldName = fieldnames(S);
length = 0;
for n = 1:numel(fieldName)
    [~, fieldLength] = S.(fieldName{n});
    length = length + fieldLength;
end
end

function [R] = readToStruct(S, ByteArray)
fieldName = fieldnames(S);
offset = 0;
for n = 1:numel(fieldName)
    [fieldType, fieldLength] = S.(fieldName{n});
    R.(fieldName{n}) = typecast(uint8(ByteArray(offset+1:offset+fieldLength)), fieldType);
    offset = offset + fieldLength;
end
end
function CS = validateChecksum(header)
h = typecast(uint8(header),'uint16');
a = uint32(sum(h));
b = uint16(sum(typecast(a,'uint16')));
CS = uint16(bitcmp(b));
end

%takes in kinematic information and outputs range, azimuth and doppler
function [H] = computeH(~, s)
posx = s(1); posy = s(2); velx = s(3); vely = s(4);
range = sqrt(posx^2+posy^2);
if posy == 0
    azimuth = pi/2;
elseif posy > 0
    azimuth = atan(posx/posy);
else
    azimuth = atan(posx/posy) + pi;
end
doppler = (posx*velx+posy*vely)/range;
H = [range azimuth doppler]';
end

function [XX, YY, ZZ, v] = gatePlot3(~, G, C, A)
%Extract the ellipsoid's axes lengths (a,b,c) and the rotation matrix (V) using singular value decomposition:
[~,D,V] = svd(A/G);

a = 1/sqrt(D(1,1));
b = 1/sqrt(D(2,2));
c = 1/sqrt(D(3,3));
v = 4*pi*a*b*c/3;

% generate ellipsoid at 0 origin
[X,Y,Z] = ellipsoid(0,0,0,a,b,c);
XX = zeros(size(X));
YY = zeros(size(X));
ZZ = zeros(size(X));
for k = 1:length(X)
    for j = 1:length(X)
        point = [X(k,j) Y(k,j) Z(k,j)]';
        P = V * point;
        XX(k,j) = P(1)+C(1);
        YY(k,j) = P(2)+C(2);
        ZZ(k,j) = P(3)+C(3);
    end
end
end

function [maxDim] = getDim(~, G, C, A)
%Extract the ellipsoid's axes lengths (a,b,c) and the rotation matrix (V) using singular value decomposition:
[~,D,V] = svd(A/G);

%ellipsoid's axes lengths (a,b,c)
a = 1/sqrt(D(1,1));
b = 1/sqrt(D(2,2));
c = 1/sqrt(D(3,3));

maxDim = max([a,b]); %outputs in essence the largest error component

end

function [y] = pow2roundup (x)
y = 1;
while x > y
    y = y * 2;
end
end

function h = circle(ax, x,y,r)
d = r*2;
px = x-r;
py = y-r;
dim = [px py d d];
h = rectangle(ax, 'Position',dim,'Curvature',[1,1], 'LineWidth',3);
daspect([1,1,1])
end

function h = updateCenter(x,y,r,offset)
d = r*2;
px = x-r;
py = y-r;
h = [px py d d];
end


function close_main()
%helpdlg('Saving and closing');
open_port = instrfind('Type','serial','Status','open');
for i=1:length(open_port)
    fclose(open_port(i));
    delete(open_port(i));
end
clear all
delete(findobj('Tag', 'mainFigure'));

end


function mypreview_fcn(obj,event,himage)
% Example update preview window function.

% Display image data.
himage.CData = fliplr(event.Data);
end

function [resInd] = getWidestFOV(resList)
maxR = 1;
resInd = 1;
for i=1:length(resList)
    ss = strsplit(resList{i},'x');
    imWidth = str2num(ss{1});
    imHeight = str2num(ss{2});
    r = imWidth/imHeight;
    if (r>maxR)
        maxR = r;
        resInd = i;
    end
end
end



