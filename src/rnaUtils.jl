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
        testInputs     = inputs[:, crossValidationIndices.==numFold];
        trainTargets   = targets[:, crossValidationIndices.!=numFold];
        testTargets    = targets[:, crossValidationIndices.==numFold];

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
        metricsInFold[numFold,:] = mean(metricsInIt, dims=1);
    end;
    metrics = mean(metricsInFold, dims=1);
    return rnaLayersSize, metricsInFold, metrics;
end;

function trainAllRNA(inputs::Array{Float64,2}, targets::Array{Bool,2}, parameters::Array{String, 1}; trainIterations::Int64=50, numFolds::Int64=10, maxCycle::Int64=1000, earlyStoppingEpochs::Int64=100, minLoss::Float64=0.0, learningRate::Float64=0.01, rnaLayers::Array{Array{Int64,1},1}=[[-1]])
    results = Array{Float64,2}[];
    resultsByFold = [];
    topologies = [];
    for array in rnaLayers
        if (length(rnaLayers) == 1 && array == [-1])
            for i in 1:8
                layer = [i];
                rnaLayersSize, dataByFold, data = bestRnaWithTopology(layer, inputs, targets, parameters, trainIterations=trainIterations, numFolds=numFolds, maxCycle=maxCycle, earlyStoppingEpochs=earlyStoppingEpochs, minLoss=minLoss, learningRate=learningRate);
                push!(results, data);
                push!(resultsByFold, dataByFold);
                push!(topologies, rnaLayersSize);
                for j in 0:3
                    layer = [i, j*2+1];
                    rnaLayersSize, dataByFold, data = bestRnaWithTopology(layer, inputs, targets, parameters, trainIterations=trainIterations, numFolds=numFolds, maxCycle=maxCycle, earlyStoppingEpochs=earlyStoppingEpochs, minLoss=minLoss, learningRate=learningRate);
                    push!(results, data);
                    push!(resultsByFold, dataByFold);
                    push!(topologies, rnaLayersSize);
                end;
            end;
        elseif issubset([0],array)
            println("La topologia introducida no es valida.");
        elseif (length(array) > 0)
            rnaLayersSize, dataByFold, data = bestRnaWithTopology(array, inputs, targets, parameters, trainIterations=trainIterations, numFolds=numFolds, maxCycle=maxCycle, earlyStoppingEpochs=earlyStoppingEpochs, minLoss=minLoss, learningRate=learningRate);

            push!(results, data);
            push!(resultsByFold, dataByFold);
            push!(topologies, rnaLayersSize);
        end;
    end;
    return topologies, resultsByFold, results;
end;

function getOrderByMetric(results, metric)
    tmp = [];
    for e in results
        push!(tmp,e[metric]);
    end;
    datos = deepcopy(tmp);
    final = [];

    while length(tmp) > 0
        toAdd = findmax(tmp)[1];
        toRem = findmax(tmp)[2];
        toAddPos = findall(x->x==toAdd, datos)[1]
        push!(final, toAddPos)
        deleteat!(tmp, toRem)
    end;
    return final;
end;


function getNBests(n::Int64, topologies, resultsByFold, globalResults, eMetrics, metrics)
    finalTopos = [];
    finalResultsByFold = [];
    finalResults = [];
    tmpOrder = Array{Any,1}(undef, length(eMetrics));
    i = 1;
    for el in eMetrics
        pos = findall(x->x==el, metrics);
        order = getOrderByMetric(globalResults, pos[1]);
        
        tmpOrder = order;
        i += 1;
    end;

    i=1;
    while i <= n && length(tmpOrder) >= i

        push!(finalTopos, topologies[tmpOrder[i]]);
        push!(finalResults, globalResults[tmpOrder[i]]);
        if (length(resultsByFold) > 0)
            push!(finalResultsByFold, resultsByFold[tmpOrder[i]]);
        end;

        i += 1;
    end;
    return finalTopos, finalResultsByFold, finalResults;
end;

function getNBestsFromFile(n::Int64, name::String)
    recall=[];
    specificity=[];
    brecindex=[];
    bspeciindex=[];
    bbothindex=[];

    open(name) do f       
        # read till end of file
        while ! eof(f) 
    
            # read a new / next line for every iteration          
            s = readline(f)   
            if startswith(s, "\tRecall:") || startswith(s, "\tSpecificity:")
                s = split(s, ":");
                val = parse(Float64, s[2]);
                if isnan(val)
                    val = 0;
                end;
                if startswith(s[1], "\tRecall")
                    push!(recall, val);
                else
                    push!(specificity, val);
                end;
            end

        end       
    end

    tmprec = deepcopy(recall);
    tmpspec = deepcopy(specificity);

    i=1;
    while i <= n && length(recall) >= i

        val = findmax(tmprec)[1];
        in = findmax(tmprec)[2];
        push!(brecindex, findall(x->x==val, recall)[1]);
        deleteat!(tmprec, in);

        val = findmax(tmpspec)[1];
        in = findmax(tmpspec)[2];
        push!(bspeciindex, findall(x->x==val, specificity)[1]);
        deleteat!(tmpspec, in);

        i += 1;
    end;

    bbothindex = intersect(brecindex, bspeciindex);

    return recall, specificity, brecindex, bspeciindex, bbothindex;


end;