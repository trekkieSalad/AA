using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;

function crossvalidation(targets::AbstractArray{Bool,2} , k::Int64)
    indices = zeros(Int64,size(targets,1));
    for i in 1:size(targets,2)
        pindices = crossvalidation(sum(targets[:,i]), k);
        pos = findall(targets[:,i]);
        indices[pos] = pindices;
        
    end;
    return indices;
end;

crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64) = crossvalidation(oneHotEncoding(targets), k);


# -------------------------------------------------------------------------
# Código de prueba:

seed!(1);

topology = [4, 3];
learningRate = 0.01;
numMaxEpochs = 1000;
numFolds = 10;
validationRatio = 0.2;
maxEpochsVal = 6;
numRepetitionsAANTraining = 50;

# Cargamos el dataset
dataset = readdlm("fonts/iris.data",',');
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = dataset[:,5];

normalizeMinMax!(inputs);

crossValidationIndices = crossvalidation(targets, numFolds);
println(crossValidationIndices)
targets=oneHotEncoding(targets);

testAccuracies = Array{Float64,1}(undef, numFolds);
testF1         = Array{Float64,1}(undef, numFolds);

# Para cada fold, entrenamos
for numFold in 1:numFolds

    # Dividimos los datos en entrenamiento y test
    local trainingInputs, testInputs, trainingTargets, testTargets;
    trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
    testInputs        = inputs[crossValidationIndices.==numFold,:];
    trainingTargets   = targets[crossValidationIndices.!=numFold,:];
    testTargets       = targets[crossValidationIndices.==numFold,:]
    
    testAccuraciesEachRepetition = Array{Float64,1}(undef, numRepetitionsAANTraining);
    testF1EachRepetition         = Array{Float64,1}(undef, numRepetitionsAANTraining);

    for numTraining in 1:numRepetitionsAANTraining

        
        local trainingIndices, validationIndices;
        (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*size(trainingInputs,1)/size(inputs,1));
        
        local ann;
        ann, = trainClassANN(topology,
            (trainingInputs[trainingIndices,:], trainingTargets[trainingIndices,:]),
            (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
            (testInputs, testTargets),
            maxEpochs=numMaxEpochs, minLoss=0.0, learningRate=learningRate, maxEpochsVal=maxEpochsVal);

        # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
        (acc, _, _, _, _, _, F1, _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

        # Almacenamos las metricas de este entrenamiento
        testAccuraciesEachRepetition[numTraining] = acc;
        testF1EachRepetition[numTraining]         = F1;

    end;

    # Almacenamos las 2 metricas que usamos en este problema
    testAccuracies[numFold] = mean(testAccuraciesEachRepetition);
    testF1[numFold]         = mean(testF1EachRepetition);

    println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

end;

println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));
