function getRNA(num_attributes::Int64, layers_size::Array{Int64,1}, num_classes::Int64)
    rna = Chain();
    rna = Chain(rna..., Dense(num_attributes, layers_size[1], σ));

    for i = 2:length(layers_size)
        rna = Chain(rna..., Dense(layers_size[i-1], layers_size[i], σ));
    end;

    if num_classes > 2
        rna = Chain(rna..., Dense(layers_size[length(layers_size)], num_classes, identity));
        rna = Chain(rna..., softmax);
    else
        rna = Chain(rna..., Dense(layers_size[length(layers_size)], 1, σ));
    end;

    return rna
end;


function trainRNA(rnaLayersSize::Array{Int64,1}, inputs::Array{Array{Float64,2},1}, targets::Array{Array{Bool,2},1}; maxCycle::Int64=1000, earlyStoppingEpochs::Int64=100, minLoss::Float64=0.0, learningRate::Float64=0.01)
    @assert(size(inputs,2) == size(targets, 2));

    trainInputs = inputs[1];
    valInputs = inputs[2];
    testInputs = inputs[3];

    trainTargets = targets[1];
    valTargets = targets[2];
    testTargets = targets[3];

    rna = getRNA(size(inputs[1],1), rnaLayersSize, size(targets[1],1));

    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(rna(x),y) : Losses.crossentropy(rna(x),y);

    cycle = 1;
    earlyStop = 0;
    lastvaloss = Inf;
    trainingLoss = Float64[];
    testLoss = Float64[];
    validationLoss = Float64[];
    bestRNA = deepcopy(rna);

    
    push!(trainingLoss, loss(trainInputs, trainTargets));
    push!(testLoss, loss(testInputs, testTargets));
    push!(validationLoss, loss(valInputs, valTargets));

    while (cycle < maxCycle && trainingLoss[length(trainingLoss)] > minLoss && earlyStop < earlyStoppingEpochs)
        Flux.train!(loss, params(rna), [(trainInputs, trainTargets)], ADAM(learningRate));
        valoss = loss(valInputs, valTargets)
        push!(trainingLoss, loss(trainInputs, trainTargets));
        push!(testLoss, loss(testInputs, testTargets));
        push!(validationLoss, valoss);

        if (valoss < lastvaloss)
            bestRNA = deepcopy(rna);
            lastvaloss = valoss;
            earlyStop = 0;
        else
            earlyStop += 1;
        end;

        cycle += 1;        
    end;


    return trainingLoss, testLoss, validationLoss, bestRNA;
end;