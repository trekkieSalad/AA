using ScikitLearn
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

#--------------------------------------------------------------------------
#                         BUILDWITHTRANSFER
#--------------------------------------------------------------------------

function buildClassANN(numInputs::Int64, topology::AbstractArray{<:Int,1}, numOutputs::Int64; fun=σ)
    ann=Chain();
    numInputsLayer = numInputs;
    for numOutputLayers = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputLayers, fun));
        numInputsLayer = numOutputLayers;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, fun));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;

#--------------------------------------------------------------------------
#                         TRAINWITHTRANSFER
#--------------------------------------------------------------------------

function trainClassANN(topology::Array{Int64,1},train::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},val::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},test::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1, maxEpochsVal::Int64=10, transfer=σ)

    trainingInputs = train[1];
    trainingTargets = train[2];
    validationInputs = val[1];
    validationTargets = val[2];
    testInputs = test[1];
    testTargets = test[2];

    withVal = (size(validationInputs,1)>0) ? true : false;

    @assert(size(trainingInputs,1)==size(trainingTargets,1));
    @assert(size(validationInputs,1)==size(validationTargets,1));
    @assert(size(testInputs,1)==size(testTargets,1));
    @assert(size(trainingInputs,2)==size(validationInputs,2)==size(testInputs,2));
    @assert(size(trainingTargets,2)==size(validationTargets,2)==size(testTargets,2));
    
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2),fun=transfer);
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    
    trainingLosses = Float64[];
    trainingAccuracies = Float64[];
    validationLosses = Float64[];
    validationAccuracies = Float64[];
    testLosses = Float64[];
    testAccuracies = Float64[];

    numEpoch = 0;

    function metricsVal()
        
        Loss   = [loss(trainingInputs', trainingTargets'),loss(testInputs', testTargets'), loss(validationInputs', validationTargets')];
        
        trainingAcc   = accuracy(Array{Float64,2}(ann(trainingInputs')'),   trainingTargets);
        validationAcc = accuracy(Array{Float64,2}(ann(validationInputs')'), validationTargets);
        testAcc       = accuracy(Array{Float64,2}(ann(testInputs')'),       testTargets);
        
        return (Loss, [trainingAcc, testAcc, validationAcc])
    end;

    function metrics()
        
        Loss   = [loss(trainingInputs', trainingTargets'), loss(testInputs', testTargets')];
        
        trainingAcc   = accuracy(Array{Float64,2}(ann(trainingInputs')'),   trainingTargets);
        testAcc       = accuracy(Array{Float64,2}(ann(testInputs')'),       testTargets);
        
        return (Loss, [trainingAcc, testAcc])
    end;

    (allLosses, Acc) = withVal ? metricsVal() : metrics();
    
    push!(trainingLosses,       allLosses[1]);
    push!(trainingAccuracies,   Acc[1]);
    push!(testLosses,           allLosses[2]);
    push!(testAccuracies,       Acc[2]);

    if withVal
        push!(validationLosses,     allLosses[3]);
        push!(validationAccuracies, Acc[3]);
        bestValidationLoss = allLosses[3];
    end;

    numEpochsValidation = 0; 
    bestANN = deepcopy(ann);

    while (numEpoch<maxEpochs) && (allLosses[1]>minLoss) && (numEpochsValidation<maxEpochsVal)
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));
        numEpoch += 1;

        (allLosses, Acc) = withVal ? metricsVal() : metrics();

        push!(trainingLosses,       allLosses[1]);
        push!(trainingAccuracies,   Acc[1]);
        push!(testLosses,           allLosses[2]);
        push!(testAccuracies,       Acc[2]);

        if withVal
            push!(validationLosses,     allLosses[3]);
            push!(validationAccuracies, Acc[3]);
            if (allLosses[3]<bestValidationLoss)
                bestValidationLoss = allLosses[3];
                numEpochsValidation = 0;
                bestANN = deepcopy(ann);
            else
                numEpochsValidation += 1;
            end;
        end;        
    end;
    return (bestANN, [trainingLosses, validationLosses, testLosses], [trainingAccuracies, validationAccuracies, testAccuracies]);
end;

#--------------------------------------------------------------------------
#                         MODELCROSSVALIDATION
#--------------------------------------------------------------------------

function modelCrossValidation(tipo::Symbol, modelHyperparameters::Dict, inputs::Array{Float64,2}, targets::Array{Any,1}, numFolds::Int64)

    @assert(size(inputs,1)==length(targets));    
    classes = unique(targets);

    if tipo==:ANN
        targets = oneHotEncoding(targets, classes);
    end;

    crossValidationIndices = crossvalidation(size(inputs,1), numFolds);

    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    for numFold in 1:numFolds

        trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
        testInputs        = inputs[crossValidationIndices.==numFold,:];
        trainingTargets   = targets[crossValidationIndices.!=numFold,:];
        testTargets       = targets[crossValidationIndices.==numFold,:];

        if (tipo==:SVM) || (tipo==:DecisionTree) || (tipo==:kNN)

            if tipo==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
            elseif tipo==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif tipo==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;

            model = fit!(model, trainingInputs, trainingTargets);

            testOutputs = predict(model, testInputs);
            
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);

        else
            @assert(tipo==:ANN);

            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);

            for numTraining in 1:modelHyperparameters["numExecutions"]

                (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                
                ann, = trainClassANN(modelHyperparameters["topology"],
                    (trainingInputs[trainingIndices,:],trainingTargets[trainingIndices,:]),
                    (trainingInputs[validationIndices,:],trainingTargets[validationIndices,:]),
                    (testInputs,testTargets);
                    maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

            end;

            acc = mean(testAccuraciesEachRepetition);
            F1  = mean(testF1EachRepetition);

        end;

        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;
    end;

    println(tipo, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println(tipo, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));

end;
