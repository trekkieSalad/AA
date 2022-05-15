function normalizeInputs(inputs::Array{Float64,2})
    min = minimum(inputs, dims = 2);
    max = maximum(inputs, dims = 2);
    inputs .-= min;
    inputs ./= (max .- min);
end;

function targetToBool(target::Array{Float32,2}, threshold::Float64=0.5)
    return Array{Bool,2}(target.>=threshold);  
end;

function dataframeFromCSV( name::String )
    return DataFrame(CSV.File(name, header=false));
end;

function dataFromDataframe( dataframe::DataFrame )
    data = Matrix(dataframe[:,1:4])';
    return [normalizeInputs(convert(Array{Float64,2}, data)); dataframe[:,5]'];
end;

function inputsFromDataframe( dataframe::DataFrame )
    data = Matrix(dataframe[:,1:4])';
    return normalizeInputs(convert(Array{Float64,2}, data));
end;

function otherInputsFromDataframe( dataframe::DataFrame )
    data = Matrix(dataframe[:,1:4]);
    return data;
end;

function outputsFromDataframe( dataframe::DataFrame )
    return Array{Bool,2}(dataframe[:,5]');
end;
function otherOutputsFromDataframe( dataframe::DataFrame )
    return dataframe[:,5];
end;

dataFromCSV( name::String ) = dataFromDataframe( dataframeFromCSV( name ) );
inputsFromCSV( name::String ) = inputsFromDataframe( dataframeFromCSV( name ) );
outputsFromCSV( name::String ) = outputsFromDataframe( dataframeFromCSV( name ) );
otherInputsFromCSV( name::String ) = otherInputsFromDataframe( dataframeFromCSV( name ) );
otherOutputsFromCSV( name::String ) = otherOutputsFromDataframe( dataframeFromCSV( name ) );

function resultsToFile(results, resultsByFold, topologies, metrics, name)

    file = open(name, "w");

    for i in 1:length(results)
        write(file, "Topology:\n\t" * join(topologies[i]) * "\n\n");
        write(file, "Metrics:\n");
        for j in 1:length(metrics)
            write(file, "\t" * metrics[j] * ": " * join(results[i][j]) * "\n");
        end;
        write(file, "\n----------------------------------------\n");

        
    end;

    close(file);
end;