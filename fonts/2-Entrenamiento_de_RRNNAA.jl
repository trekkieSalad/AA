using Statistics
using Flux
using Flux.Losses
using DelimitedFiles;

#--------------------------------------------------------------------------
#                         ONEHOTENCODING
#--------------------------------------------------------------------------

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    @assert(all([in(value, classes) for value in feature]));
    numClasses = length(classes);
    @assert(numClasses>1)
    
    if (numClasses==2)
        oneHot = Array{Bool,2}(undef, size(feature,1), 1);
        oneHot[:,1] .= (feature.==classes[1]);
    else
        oneHot = Array{Bool,2}(undef, size(feature,1), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature::AbstractArray{<:Any,1}, unique(feature));
oneHotEncoding(feature::AbstractArray{Bool,1}) = feature;

#--------------------------------------------------------------------------
#                         NORMALIZE
#--------------------------------------------------------------------------

calculateMinMaxNormalizationParameters(data::AbstractArray{<:Real,2}; inRow=true) =
    ( minimum(data, dims=(inRow ? 1 : 2)), maximum(data, dims=(inRow ? 1 : 2)) );

function normalizeMinMax!(data::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}; inRow=true)
    min = normalizationParameters[1];
    max = normalizationParameters[2];
    data .-= min;
    data ./= (max .- min);
    if (inRow)
        data[:, vec(min.==max)] .= 0;
    else
        data[vec(min.==max), :] .= 0;
    end
end;

function normalizeMinMax(data::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}; inRow=true)
    newDataset = copy(data);
    normalizeMinMax!(newDataset, normalizationParameters; inRow=inRow);
    return newDataset;
end;

normalizeMinMax!(data::AbstractArray{<:Real,2}; inRow=true) = normalizeMinMax!(data, calculateMinMaxNormalizationParameters(data; inRow=inRow); inRow=inRow);
normalizeMinMax(data::AbstractArray{<:Real,2}; inRow=true) = normalizeMinMax(data, calculateMinMaxNormalizationParameters(data; inRow=inRow); inRow=inRow);
    

calculateZeroMeanNormalizationParameters(data::AbstractArray{<:Real,2}; inRows=true) =
    ( mean(data, dims=(inRows ? 1 : 2)), std(data, dims=(inRows ? 1 : 2)) );

function normalizeZeroMean!(data::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}; inRows=true)
    avg  = normalizationParameters[1];
    stnd = normalizationParameters[2];
    data .-= avg;
    data ./= stnd;
    
    if (inRows)
        data[:, vec(stnd.==0)] .= 0;
    else
        data[vec(stnd.==0), :] .= 0;
    end
end;

function normalizeZeroMean(data::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}; inRows=true)
    newDataset = copy(data);
    normalizeZeroMean!(newDataset, normalizationParameters; inRows=inRows);
    return newDataset;
end;

normalizeZeroMean!(data::AbstractArray{<:Real,2}; inRows=true) = normalizeZeroMean!(data, calculateZeroMeanNormalizationParameters(data; inRows=inRows); inRows=inRows);
normalizeZeroMean(data::AbstractArray{<:Real,2}; inRows=true) = normalizeZeroMean(data, calculateZeroMeanNormalizationParameters(data; inRows=inRows); inRows=inRows);

#--------------------------------------------------------------------------
#                         CLASSIFYOUTPUTS
#--------------------------------------------------------------------------

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Float64=0.5)
    numOutputs = size(outputs, 2);

    if numOutputs==1
        return convert(Array{Bool,2}, outputs.>=threshold);
    else
        (_,indicesMaxEachInstance) = findmax(outputs, dims= 2);
        outputsBoolean = Array{Bool,2}(falses(size(outputs)));
        outputsBoolean[indicesMaxEachInstance] .= true;
        @assert(all(sum(outputsBoolean, dims=2).==1));
        return outputsBoolean;
    end;
end;

#--------------------------------------------------------------------------
#                         ACCURACY
#--------------------------------------------------------------------------

accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs.==targets);

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        classComparison = targets .== outputs
        correctClassifications = all(classComparison, dims=2)
        return mean(correctClassifications)
    end;
end;

accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Float64=0.5) = accuracy(AbstractArray{Bool,1}(outputs.>=threshold), targets);

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2})
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs), targets);
    end;
end;

# Añado estas funciones porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
# Con estas funciones se pueden usar indistintamente matrices de Float32 o Float64
accuracy(outputs::Array{Float32,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = accuracy(Float64.(outputs), targets; threshold=threshold);
accuracy(outputs::Array{Float32,2}, targets::Array{Bool,2}; dataInRows::Bool=true)  = accuracy(Float64.(outputs), targets; dataInRows=dataInRows);

#--------------------------------------------------------------------------
#                         BUILDRNA
#--------------------------------------------------------------------------

function buildClassANN(numInputs::Int64, topology::AbstractArray{<:Int,1}, numOutputs::Int64)
    ann=Chain();
    numInputsLayer = numInputs;
    for numOutputLayers = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputLayers, σ));
        numInputsLayer = numOutputLayers;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;

#--------------------------------------------------------------------------
#                         TRAINRNA
#--------------------------------------------------------------------------


function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; maxEpochs::Int64=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    inputs = dataset[1];
    targets = dataset[2];
    
    @assert(size(inputs,1)==size(targets,1));

    ann = buildClassANN(size(inputs,2), topology, size(targets,2));
    
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    trainingLosses = Float64[];
    trainingAccuracies = Float64[];
    
    numEpoch = 0;
    trainingLoss = Inf;

    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)
        
        Flux.train!(loss, params(ann), [(inputs', targets')], ADAM(learningRate));
        numEpoch += 1;
        trainingLoss = loss(inputs', targets');
        outputs = ann(inputs');
        trainingAccuracy = accuracy(Array{Float64,2}(outputs'), targets);
        push!(trainingLosses, trainingLoss);
        push!(trainingAccuracies, trainingAccuracy);
    end;

    println("Total epochs ", numEpoch, ": best accuracy: ", 100*maximum(trainingAccuracies), " %");
    return (ann, trainingLosses, trainingAccuracies);
end;


#--------------------------------------------------------------------------
#                         EJECUCION
#--------------------------------------------------------------------------

# Por norma general no se observan variaciones reseñables entre el uso de datos normalizados y no 
# normalizados (salvo en algunas situaciones en que los datos normalizados arrojan mejores resultados)
# 
# Por otro lado la metrica con la que mejores resultados obtuvimos fue la [3,2]

topology = [8];
learningRate = 0.01;
numMaxEpochs = 1000;


dataset = readdlm("fonts/iris.data",',');
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);
data = (inputs, targets);

(ann, trainingLosses, trainingAccuracies) = trainClassANN(topology, data; maxEpochs=numMaxEpochs, learningRate=learningRate);
normalizeMinMax!(inputs);
(ann, trainingLosses, trainingAccuracies) = trainClassANN(topology, data; maxEpochs=numMaxEpochs, learningRate=learningRate);


