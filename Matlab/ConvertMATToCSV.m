load('C:\Users\hasna\Documents\GitHub\OccupancyDetection\Data\2peoplecentroidData.mat');
data = data(101:end);

%loop through data cell array
centroidArray = [];
for frameNo = 1:length(data)
    frame = data{frameNo};
    for row = 1:size(frame,1)
        
    end
        
end
% 
% tableFile = table([centroidArray]);
% writetable(tableFile, 'testCSV.csv');