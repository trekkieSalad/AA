using Random

#--------------------------------------------------------------------------
#                         HOLDOUT
#--------------------------------------------------------------------------


function holdOut(N::Int, P::Float64)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N);
    toTrain = Int(round(N*(1-P)));
    return (indices[1:toTrain], indices[toTrain+1:end]);
end

function holdOut(N::Int, Pval::Float64, Ptest::Float64)
    @assert ((Pval>=0.) & (Pval<=1.));
    @assert ((Ptest>=0.) & (Ptest<=1.));
    @assert ((Pval+Ptest)<=1.);
    
    (trainval, test) = holdOut(N, Ptest);
    (train, val) = holdOut(length(trainval), Pval*N/length(trainval))
    return (trainval[train], trainval[val], test);
end;

#--------------------------------------------------------------------------
#                         TRAINVALIDATION
#--------------------------------------------------------------------------

function trainClassANN(topology::Array{Int64,1},train::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},val::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},test::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1, maxEpochsVal::Int64=10)

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
    
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2));
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
        
        println("Epoch ", numEpoch, ": Training accuracy: ", 100*trainingAcc, " %, validation accuracy: ", 100*validationAcc, " %, test accuracy: ", 100*testAcc, " %");
        return (Loss, [trainingAcc, testAcc, validationAcc])
    end;

    function metrics()
        
        Loss   = [loss(trainingInputs', trainingTargets'), loss(testInputs', testTargets')];
        
        trainingAcc   = accuracy(Array{Float64,2}(ann(trainingInputs')'),   trainingTargets);
        testAcc       = accuracy(Array{Float64,2}(ann(testInputs')'),       testTargets);
        
        println("Epoch ", numEpoch, ": Training accuracy: ", 100*trainingAcc, " %, test accuracy: ", 100*testAcc, " %");
        return (Loss, [trainingAcc, validationAcc, testAcc])
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


# -------------------------------------------------------------------------

topology = [3, 2];
learningRate = 0.01;
numMaxEpochs = 1000;
validationRatio = 0.2;
testRatio = 0.2;
maxEpochsVal = 6;


dataset = readdlm("fonts/iris.data",',');
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);

normalizeMinMax!(inputs);
(trainingIndices, validationIndices, testIndices) = holdOut(size(inputs,1), validationRatio, testRatio);

# Dividimos los datos
train    = (inputs[trainingIndices,:], targets[trainingIndices,:]);
val      = (inputs[validationIndices,:], targets[validationIndices,:]);
test     = (inputs[testIndices,:], targets[testIndices,:]);

(ann, trainingLosses, trainingAccuracies) = trainClassANN(topology,train,val,test,maxEpochs=500, minLoss=0.01, learningRate=0.1, maxEpochsVal=10);