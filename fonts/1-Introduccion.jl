using FileIO;
using DelimitedFiles;
using Statistics;

function codCategorica(feature::Array{Any,1}, classes::Array{Any,1})
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

codCategorica(feature::Array{Any,1}) = codCategorica(feature::Array{Any,1}, unique(feature));
codCategorica(feature::Array{Bool,1}) = feature;

function valuesOf(m::Array{Float64,2})
    min = minimum(m, dims=1);
    max = maximum(m, dims=1);
    media = mean(m, dims=1);
    desviacion = std(m, dims=1);
    return min, max, media, desviacion;
end;

function normalizeData(inputs::Array{Float64,2}, inRow=false)
    min = minimum(inputs, dims = 1+inRow);
    max = maximum(inputs, dims = 1+inRow);
    inputs .-= min;
    inputs ./= (max .- min);

    if (inRow)
        inputs[vec(min.==max), :] .= 0;
    else
        inputs[:, vec(min.==max)] .= 0;
    end
    return inputs;
end;

# Haciendo matrices de entradas y salidas

dataset = readdlm("fonts/iris.data",',');

inputs = Float64.(dataset[:,1:4]);
targets = codCategorica(dataset[:,5]);
println("Tamaño de las entradas: ", size(inputs,1), "x", size(inputs,2), " de tipo ", typeof(inputs));
println("Tamaño de los targets: ", size(targets,1), "x", size(targets,2), " de tipo ", typeof(targets));


normalizedInputs = normalizeData(inputs);
m = valuesOf(normalizedInputs);
println("Minimos de las entradas: ", m[1]);
println("Maximos de las entradas: ", m[2]);
println("Medias de las entradas: ", m[3]);
println("Desviaciones de las entradas: ", m[4]);