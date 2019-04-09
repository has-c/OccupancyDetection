clear, clc

%% SETUP SELECTIONS

% 1. sceneRun needs to be specied.
% Default is to use GUI to setup scene
% To programmatically setup the scene follow the example in Prgm_2box.
 
% Scene Run choice: {'GUI_Setup' | 'Prgm_2box' | 'Prgm_MaxFOV','My_Scene'};
sceneRun = 'GUI_Setup';




%% Setup tracking scene

% Enables setting parameters by GUI
if(strcmp(sceneRun,'GUI_Setup'))
    
    % Call setup GUI
    hSetup = setup();
    close(hSetup.figure1);
   
    hDataSerialPort = hSetup.hDataSerialPort; %initializes objects
    hControlSerialPort = hSetup.hControlSerialPort;
  
    loadCfg = 0; %disabled because loaded in GUI
end



%% Serial setup
if (~strcmp(sceneRun,'GUI_Setup'))
    %Configure data UART port with input buffer to hold 100+ frames 
    hDataSerialPort = configureDataSport(dataSerialPort, 65536);
    
    %Send Configuration Parameters to IWR16xx
    if(loadCfg)
        mmwDemoCliPrompt = char('mmwDemo:/>');
        hControlSerialPort = configureControlPort(controlSerialPort);
        %Send CLI configuration to IWR16xx
        fprintf('Sending configuration from %s file to IWR16xx ...\n', configurationFileName);
        for k=1:length(cliCfg)
            fprintf(hControlSerialPort, cliCfg{k});
            fprintf('%s\n', cliCfg{k});
            echo = fgetl(hControlSerialPort); % Get an echo of a command
            done = fgetl(hControlSerialPort); % Get "Done" 
            prompt = fread(hControlSerialPort, size(mmwDemoCliPrompt,2)); % Get the prompt back 
        end
        fclose(hControlSerialPort);
        delete(hControlSerialPort);
    end
end

%% Init variables
trackerRun = 'Target';
colors='brgcm';
labelTrack = 0;

%sensor parameters
%sensor.rangeMax = 6;
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
                offset = offset + tlvHeaderLengthInBytes;
                valueLength = tlvLength - tlvHeaderLengthInBytes; %calculate length of the actual TLV data 
                switch(tlvType) %depending on TLV Type run a particular portion of code 
                    
                    case 6  % Point Cloud TLV
                        numInputPoints = valueLength/pointLengthInBytes; %calculate number of avaliable points => total TLV frame length / length of one piece of point data
                        if(numInputPoints > 0) %actually have some points to parse                        
                            % Get Point Cloud from the sensor
                            p = typecast(uint8(rxData(offset+1: offset+valueLength)),'single'); %get all avaliable point cloud data from the sensor

                            pointCloud = reshape(p,4, numInputPoints); %form point cloud, resultant matrix is 4 x numInputPoints in size
%                            pointCloud(2,:) = pointCloud(2,:)*pi/180;

                            posAll = [pointCloud(1,:).*sin(pointCloud(2,:)); pointCloud(1,:).*cos(pointCloud(2,:))]; %calculate x y positions 
                            snrAll = pointCloud(4,:);

                            % Remove out of Range, Behind the Walls, out of FOV points
                            inRangeInd = (pointCloud(1,:) > 1) & (pointCloud(1,:) < 6) & ...
                                (pointCloud(2,:) > -50*pi/180) &  (pointCloud(2,:) < 50*pi/180) & ...
                                (posAll(1,:) > scene.areaBox(1)) & (posAll(1,:) < (scene.areaBox(1) + scene.areaBox(3))) & ...
                                (posAll(2,:) > scene.areaBox(2)) & (posAll(2,:) < (scene.areaBox(2) + scene.areaBox(4)));
                            pointCloudInRange = pointCloud(:,inRangeInd);
                            posInRange = posAll(:,inRangeInd);
%{
                            % Clutter removal
                            staticInd = (pointCloud(3,:) == 0);        
                            clutterInd = ismember(pointCloud(1:2,:)', clutterPoints', 'rows');
                            clutterInd = clutterInd' & staticInd;
                            clutterPoints = pointCloud(1:2,staticInd);
                            pointCloud = pointCloud(1:3,~clutterInd);
%}
                            numOutputPoints = size(pointCloud,2);                          
                        end                        
                        offset = offset + valueLength;
                                            
                    case 7
                        % Target List TLV
                        numTargets = valueLength/targetLengthInBytes;                        
                        TID = zeros(1,numTargets);
                        S = zeros(6, numTargets);
                        EC = zeros(9, numTargets);
                        G = zeros(1,numTargets);                        
                        for n=1:numTargets
                            TID(n)  = typecast(uint8(rxData(offset+1:offset+4)),'uint32');      %1x4=4bytes
                            S(:,n)  = typecast(uint8(rxData(offset+5:offset+28)),'single');     %6x4=24bytes
                            EC(:,n) = typecast(uint8(rxData(offset+29:offset+64)),'single');    %9x4=36bytes
                            G(n)    = typecast(uint8(rxData(offset+65:offset+68)),'single');    %1x4=4bytes
                            offset = offset + 68;
                        end
                        
                    case 8
                        % Target Index TLV
                        numIndices = valueLength/indexLengthInBytes;
                        mIndex = typecast(uint8(rxData(offset+1:offset+numIndices)),'uint8');
                        offset = offset + valueLength;
                end
            end
        end
       
        if(numInputPoints == 0)
            numOutputPoints = 0;
            pointCloud = single(zeros(4,0));
            posAll = [];
            posInRange = [];  
        end
        if(numTargets == 0)
            TID = [];
            S = [];
            EC = [];
            G = [];
        end
        
        fHist(frameNum).numInputPoints = numInputPoints;
        fHist(frameNum).numOutputPoints = numOutputPoints;    
        fHist(frameNum).numTargets = numTargets;
        fHist(frameNum).pointCloud = pointCloud;
        fHist(frameNum).targetList.numTargets = numTargets;
        fHist(frameNum).targetList.TID = TID;
        fHist(frameNum).targetList.S = S;
        if(~optimize)
            fHist(frameNum).targetList.EC = EC;
        end
        fHist(frameNum).targetList.G = G;
        fHist(frameNum).indexArray = mIndex;
       
        % Plot pointCloud
        fHist(frameNum).benchmarks(2) = 1000*toc(frameStart);
   
        if(get(hRbPause, 'Value') == 1)
            pause(0.01);
            continue;
        end
        
        % Delete previous points
        if(ishandle(hPlotCloudHandleAll))
            delete(hPlotCloudHandleAll);
        end
        if(ishandle(hPlotCloudHandleOutRange))
            delete(hPlotCloudHandleOutRange);
        end
        if(ishandle(hPlotCloudHandleClutter))
            delete(hPlotCloudHandleClutter);
        end
        if(ishandle(hPlotCloudHandleStatic))
            delete(hPlotCloudHandleStatic);
        end
        if(ishandle(hPlotCloudHandleDynamic))
            delete(hPlotCloudHandleDynamic);
        end

        if(size(posAll,2))
            % Plot all points
            if(snrAll*10 > 0)
                if(~optimize)
                    hPlotCloudHandleAll = scatter(trackingAx, posAll(1,:), posAll(2,:),'.k','SizeData',snrAll*10);
                else
                    hPlotCloudHandleAll = plot(trackingAx, posAll(1,:), posAll(2,:),'.k');
                end
            else
                reason = 'SNR value is wrong';
                lostSync = 1;
                break;                
            end
            % Cross out out-of-Range
            if(~optimize)
                hPlotCloudHandleOutRange = plot(trackingAx, posAll(1,~inRangeInd), posAll(2,~inRangeInd), 'xr');
            end
        end
        
%{        
        if(size(posInRange,2))
            % Cross out Clutter
            hPlotCloudHandleClutter = plot(trackingAx, posInRange(1,clutterInd), posInRange(2,clutterInd), 'xk');
            % Indicate Static
            hPlotCloudHandleStatic = plot(trackingAx, posInRange(1,staticInd & ~clutterInd), posInRange(2,staticInd & ~clutterInd), 'ok');
            % Indicate Dynamic
            hPlotCloudHandleDynamic = plot(trackingAx, posInRange(1,~staticInd), posInRange(2,~staticInd), 'ob');
        end
%}        
        fHist(frameNum).benchmarks(3) = 1000*toc(frameStart);

        switch trackerRun
            case 'Target'
                if(numTargets == 0)
                    TID = zeros(1,0);
                    S = zeros(6,0);
                    EC = zeros(9,0);
                    G = zeros(1,0);
                end
        end
        
        fHist(frameNum).benchmarks(4) = 1000*toc(frameStart);
        
        if nnz(isnan(S))
            reason = 'Error: S contains NaNs';
            lostSync = 1;
            break;
        end
        if nnz(isnan(EC))
            reason = 'Error: EC contains NaNs';
            lostSync = 1;
            break;
        end
        
        tNumC = length(TID);
        peopleCountTotal = tNumC;
        peopleCountInBox = zeros(1, scene.numberOfTargetBoxes);
 
        if(size(mIndex,1)) 
            mIndex = mIndex + 1;
        end
        
        % Plot previous frame's 3D points       
        if(size(point3D,2))   
            if isempty(hPlotPoints3D)
                %hPlotPoints3D = plot(gatingAx, point3D(1,:), point3D(2,:), '.k');%, point3D(3,:),'.k');
            else
                %set(hPlotPoints3D, 'XData', point3D(1,:),'YData', point3D(2,:)); %, 'ZData', point3D(3,:));
            end
        end     
        
        for n=1:tNumC
            
            tid = TID(n)+1;            
            if(tid > maxNumTracks)
                reason = 'Error: TID is wrong';
                lostSync = 1;
                break;
            end

            if( (size(mIndex,1) > 0) && (size(mIndex,1) == size(point3D,2)) )
                tColor = colors(mod(tid,length(colors))+1);
                ind = (mIndex == tid);
                if nnz(ind)
                    %trackingHist(tid).hPlotAssociatedPoints = plot(gatingAx, point3D(1,ind), point3D(2,ind), 'o', 'color', tColor)
                    if (isempty(trackingHist(tid).hPlotAssociatedPoints) || ~ishandle(trackingHist(tid).hPlotAssociatedPoints))
                        trackingHist(tid).hPlotAssociatedPoints = plot(gatingAx, point3D(1,ind), point3D(2,ind), '.', 'MarkerSize', 10, 'color', tColor);
                    else
                        if ishandle(trackingHist(tid).hPlotAssociatedPoints)
                            set(trackingHist(tid).hPlotAssociatedPoints, 'XData', point3D(1,ind),'YData', point3D(2,ind));
                        end
                    end
                end
            end
               
            %g = G(n);
            centroid = computeH(1, S(:,n));
            ec = reshape(EC(:,n),3,3);
            if(nnz(ec)>1)
                dim = getDim(gatingAx, 1, centroid, ec);
                
                if isempty(trackingHist(tid).hMeshU)
                    trackingHist(tid).hMeshU = circle(gatingAx,S(1,n), S(2,n),dim);
                    if(labelTrack)
                        trackingHist(tid).hMeshU.UserData = text(gatingAx,S(1,n), S(2,n), char(65+mod(tid,26)),'HorizontalAlignment','center','FontUnits','normalized','FontSize',0.5/scene.maxPos(4)*0.75);
                    end
                        if(~optimize)
                        trackingHist(tid).hMeshU.EdgeColor = [0.5 0.5 0.5];
                    else
                        trackingHist(tid).hMeshU.EdgeColor = colors(mod(tid,length(colors))+1);
                    end
                    trackingHist(tid).hMeshU.FaceColor = 'none';
                else
                    %set(trackingHist(tid).hMeshU, 'XData', xU.*sin(yU),'YData',xU.*cos(yU), 'ZData', zU);
                    trackingHist(tid).hMeshU.Position = updateCenter(S(1,n), S(2,n),dim);
                    if(labelTrack)
                        trackingHist(tid).hMeshU.UserData.Position = [S(1,n), S(2,n)];
                    end
                end
            end
            
            if(~optimize)
                if(g ~= 0)
                    [xG, yG, zG, vG] = gatePlot3(gatingAx, g, centroid, ec);
                    if isempty(trackingHist(tid).hMeshG)
                        trackingHist(tid).hMeshG = mesh(gatingAx, xG.*sin(yG),xG.*cos(yG), zG);
                        trackingHist(tid).hMeshG.EdgeColor = colors(mod(tid,length(colors))+1);
                        trackingHist(tid).hMeshG.FaceColor = 'none';
                    else
                        set(trackingHist(tid).hMeshG, 'XData', xG.*sin(yG),'YData',xG.*cos(yG), 'ZData', zG);
                    end
                end
            end
            
            if(activeTracks(tid) == 0)
                activeTracks(tid) = 1;
                trackingHist(tid).tid = TID(n);
                trackingHist(tid).allocationTime = targetFrameNum;
                trackingHist(tid).tick = 1;
                trackingHist(tid).posIndex = 1;
                trackingHist(tid).histIndex = 1;
                trackingHist(tid).sHat(1,:) = S(:,n);
                trackingHist(tid).pos(1,:) = S(1:2,n);               
                trackingHist(tid).hPlotTrack = plot(trackingAx, S(1,n), S(2,n), '-', 'LineWidth', 2, 'color', colors(mod(tid,length(colors))+1));
                trackingHist(tid).hPlotCentroid = plot(trackingAx, S(1,n), S(2,n), 'o', 'color', colors(mod(tid,length(colors))+1));
            else
                activeTracks(tid) = 1;                
                trackingHist(tid).tick = trackingHist(tid).tick + 1;
                
                trackingHist(tid).histIndex = trackingHist(tid).histIndex + 1;
                if(trackingHist(tid).histIndex > 1000)
                    trackingHist(tid).histIndex = 1;
                end
                trackingHist(tid).sHat(trackingHist(tid).histIndex,:) = S(:,n);
                if(~optimize)
                    trackingHist(tid).ec(trackingHist(tid).histIndex,:) = EC(:,n);
                end
                trackingHist(tid).posIndex = trackingHist(tid).posIndex + 1;
                if(trackingHist(tid).posIndex > 100)
                    trackingHist(tid).posIndex = 1;
                end
                trackingHist(tid).pos(trackingHist(tid).posIndex,:) = S(1:2,n);

                if(trackingHist(tid).tick > 100)
                    set(trackingHist(tid).hPlotTrack, 'XData', [trackingHist(tid).pos(trackingHist(tid).posIndex+1:end,1); trackingHist(tid).pos(1:trackingHist(tid).posIndex,1)], ...
                        'YData',[trackingHist(tid).pos(trackingHist(tid).posIndex+1:end,2); trackingHist(tid).pos(1:trackingHist(tid).posIndex,2)]);
                else
                    set(trackingHist(tid).hPlotTrack, 'XData', trackingHist(tid).pos(1:trackingHist(tid).posIndex,1), ...
                        'YData',trackingHist(tid).pos(1:trackingHist(tid).posIndex,2));
                end                
                set(trackingHist(tid).hPlotCentroid,'XData',S(1,n),'YData',S(2,n));
                
                for nBoxes = 1:scene.numberOfTargetBoxes 
                    if( (S(1,n) > scene.targetBox(nBoxes,1)) && (S(1,n) < (scene.targetBox(nBoxes,1) + scene.targetBox(nBoxes,3))) && ...
                            (S(2,n) > scene.targetBox(nBoxes,2)) && (S(2,n) < (scene.targetBox(nBoxes,2)+scene.targetBox(nBoxes,4))) )
                        peopleCountInBox(nBoxes) = peopleCountInBox(nBoxes) +1;
                    end  
                end               
            end
        end
        
        iDelete = find(activeTracks == 2);
        for n=1:length(iDelete)
            ind = iDelete(n);
            delete(trackingHist(ind).hPlotTrack);
            delete(trackingHist(ind).hPlotCentroid);
            delete(trackingHist(ind).hMeshU.UserData);
            delete(trackingHist(ind).hMeshU);
            delete(trackingHist(ind).hMeshG);
            delete(trackingHist(ind).hPlotAssociatedPoints);
            trackingHist(ind).hMeshU = [];
            trackingHist(ind).hMeshG = [];           
            activeTracks(ind) = 0;
        end
        
        iReady = (activeTracks == 1);
        activeTracks(iReady) = 2;
        
        fHist(frameNum).done = 1000*toc(frameStart);
        
        string{1} = sprintf('Frame #: %d', targetFrameNum);
        string{2} = sprintf('Detection Points: %d', numOutputPoints);
        string{3} = sprintf('People Count: %d', peopleCountTotal);
        for i=1:scene.numberOfTargetBoxes
            string{3+i} = sprintf('Box %d Count: %d', i, peopleCountInBox(i));
        end
        string{3+scene.numberOfTargetBoxes+1} = sprintf('Bytes Available: %d/%d', bytesAvailable, maxBytesAvailable);

        for n=1:length(hStatGlobal)
            set(hStatGlobal(n),'String',string{n});
        end
        
        
        for nBoxes = 1:scene.numberOfTargetBoxes    
            if(peopleCountInBox(nBoxes))
                set(hTargetBoxHandle(nBoxes), 'LineWidth', 14);
            else
                set(hTargetBoxHandle(nBoxes), 'LineWidth', 4);
            end
        end

        if(getappdata(hPbExit, 'exitKeyPressed') == 1)
            if(frameNumLogged > 10000)
                fHist = [fHist(frameNum+1:end) fHist(1:frameNum)];
            else
                fHist = fHist(1:frameNum);
            end
            
            save('fhistRT.mat','fHist');
            disp('Saving data and exiting');
            close_main()
            return;
        end
        
        frameNum = frameNum + 1;
        frameNumLogged = frameNumLogged + 1;      
        if(frameNum > 10000)
            frameNum = 1;
        end
                
        point3D = [posAll; pointCloud(3,:)];
        
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

    a = 1/sqrt(D(1,1));
    b = 1/sqrt(D(2,2));
    c = 1/sqrt(D(3,3));
    
    maxDim = max([a,b]);

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



