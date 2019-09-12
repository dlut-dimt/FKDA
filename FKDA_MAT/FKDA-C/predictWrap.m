function [predictLabel, precision,t_p,probability] = predictWrap(trainData, trainLabel, testData, testLabel)
    tic
    model = fitcknn(trainData, trainLabel, 'NumNeighbors', 1, 'Standardize', 1);
    [predictLabel,probability,~] = predict(model, testData);
    precision = double(sum(predictLabel == testLabel')) / length(testLabel);
    t_p = toc;
end
