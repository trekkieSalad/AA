using Flux
using Flux.Losses
using Statistics

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


function trainRNA(rnaLayersSize::Array{Int64,1}, inputs::Array{Float64,2}, targets::Array{Bool,2}, maxCycle::Int64, minLoss::Float64=0.0, learningRate::Float64=0.01)
    @assert(size(inputs,2) == size(targets, 2));

    (trainIndex, valIndex, testIndex) = holdOut(size(inputs,2), .2, .2);
    trainInputs = inputs[:, trainIndex];
    valInputs = inputs[:, valIndex];
    testInputs = inputs[:, testIndex];

    trainTargets = targets[:, trainIndex];
    valTargets = targets[:, valIndex];
    testTargets = targets[:, testIndex];

    rna = getRNA(size(inputs,1), rnaLayersSize, size(targets,1));

    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(rna(x),y) : Losses.crossentropy(rna(x),y);

    cycle = 1;
    trainingLoss = Float64[];
    testLoss = Float64[];
    validationLoss = Float64[];

    
    push!(trainingLoss, loss(trainInputs, trainTargets));
    push!(testLoss, loss(testInputs, testTargets));
    push!(validationLoss, loss(valInputs, valTargets));

    while (cycle < maxCycle && trainingLoss[length(trainingLoss)] > minLoss )
        Flux.train!(loss, params(rna), [(trainInputs, trainTargets)], ADAM(learningRate));
        push!(trainingLoss, loss(trainInputs, trainTargets));
        push!(testLoss, loss(testInputs, testTargets));
        push!(validationLoss, loss(valInputs, valTargets));
        cycle += 1;        
    end;
    outputs = rna(inputs);

    println(targetToBool(outputs));

    println("VP: ", getVP(targetToBool(outputs), targets));
    println("VN: ", getVN(targetToBool(outputs), targets));
    println("FP: ", getFP(targetToBool(outputs), targets));
    println("FN: ", getFN(targetToBool(outputs), targets));

    println("Precision: ", accuracy(targetToBool(outputs), targets));
    println("Ratio de error: ", errorRate(targetToBool(outputs), targets));
    println("Sensibilidad: ", recall(targetToBool(outputs), targets));
    println("Especificidad: ", specificity(targetToBool(outputs), targets));
    println("VPP: ", ppv(targetToBool(outputs), targets));
    println("VPN: ", npv(targetToBool(outputs), targets));
    println("F1-SCORE: ", f1(targetToBool(outputs), targets));


    return trainingLoss, testLoss, validationLoss;
end;