crossvalidation(N::Int64, k::Int64) = shuffle!(repeat(1:k, Int64(ceil(N/k)))[1:N]);


function holdOut(N::Int, P::Float64)
    indexes = randperm(N);
    training = Int(round(N*(1-P)));
    return (indexes[1:training], indexes[training+1:end]);
end;

function holdOut(N::Int, validation::Float64, test::Float64)

    (trainValIndex, testIndex) = holdOut(N, test);
    (trainIndex, valIndex) = holdOut(length(trainValIndex), validation*N/length(trainValIndex))

    return (trainValIndex[trainIndex], trainValIndex[valIndex], testIndex);
end;

function bestRnaWithTopology(rnaLayersSize::Array{Int64,1}, inputs::Array{Float64,2}, targets::Array{Bool,2}, parameters::Array{String, 1}; trainIterations::Int64=50, numFolds::Int64=10, maxCycle::Int64=1000, earlyStoppingEpochs::Int64=100, minLoss::Float64=0.0, learningRate::Float64=0.01)
    
    crossValidationIndices = crossvalidation(size(inputs,2), numFolds);
    text = "Entrenando topologia " * string(rnaLayersSize) * ": ";
    p = Progress(numFolds*trainIterations, text);
    metrics = Array{Float64,1}(undef,length(parameters));
    metricsInFold = Array{Float64,2}(undef, numFolds, length(parameters));

    for numFold in 1:numFolds

        trainInputs    = inputs[:, crossValidationIndices.!=numFold];
        testInputs        = inputs[:, crossValidationIndices.==numFold];
        trainTargets   = targets[:, crossValidationIndices.!=numFold];
        testTargets       = targets[:, crossValidationIndices.==numFold];

        metricsInIt = Array{Any,2}(undef, trainIterations, length(parameters));

        for iteration in 1:trainIterations

            (trainIndexes, validationIndexes) = holdOut(size(trainInputs,2), .2);

            newInputs = [trainInputs[:, trainIndexes], trainInputs[:, validationIndexes], testInputs];
            newTargets = [trainTargets[:, trainIndexes], trainTargets[:, validationIndexes], testTargets];

            (trainLoss, testLoss, valLoss, rna) = trainRNA(rnaLayersSize, newInputs, newTargets, maxCycle=maxCycle, earlyStoppingEpochs=earlyStoppingEpochs, minLoss=minLoss, learningRate=learningRate);

            mymetrics = getMetrics(targetToBool(rna(testInputs)), testTargets, parameters);
            
            metricsInIt[iteration,:] = mymetrics;
            next!(p; showvalues= [(:fold, numFold) (:iteracion, iteration)]);
        end;
        #println(metricsInIt);
        metricsInFold[numFold,:] = mean(metricsInIt, dims=1);
    end;
    metrics = mean(metricsInFold, dims=1);
    println(metrics);
end;

function bestRNA(inputs::Array{Float64,2}, targets::Array{Bool,2}, parameters::Array{String, 1}; trainIterations::Int64=50, numFolds::Int64=10, maxCycle::Int64=1000, earlyStoppingEpochs::Int64=100, minLoss::Float64=0.0, learningRate::Float64=0.01, rnaLayers::Array{Array{Int64,1},1}=[[-1]])
    for array in rnaLayers 
        println(array)
        if (length(rnaLayers) == 1 && array == [-1])
            for i in 1:10
                layer = [i];
                bestRnaWithTopology(layer, inputs, targets, parameters, trainIterations=trainIterations, numFolds=numFolds, maxCycle=maxCycle, earlyStoppingEpochs=earlyStoppingEpochs, minLoss=minLoss, learningRate=learningRate);
                for j in 1:10
                    layer = [i, j];
                    bestRnaWithTopology(layer, inputs, targets, parameters, trainIterations=trainIterations, numFolds=numFolds, maxCycle=maxCycle, earlyStoppingEpochs=earlyStoppingEpochs, minLoss=minLoss, learningRate=learningRate);
                end;
            end;
        elseif issubset([0],array)
            println("La topologia introducida no es valida.");
        elseif (length(array) > 0)
            bestRnaWithTopology(array, inputs, targets, parameters, trainIterations=trainIterations, numFolds=numFolds, maxCycle=maxCycle, earlyStoppingEpochs=earlyStoppingEpochs, minLoss=minLoss, learningRate=learningRate);
        end;
    end;
end;
