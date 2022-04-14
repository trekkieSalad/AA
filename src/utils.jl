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
    data = Matrix(dataframe[:,1:2])';
    return [normalizeInputs(convert(Array{Float64,2}, data)); dataframe[:,3]'];
end;

function inputsFromDataframe( dataframe::DataFrame )
    data = Matrix(dataframe[:,1:2])';
    return normalizeInputs(convert(Array{Float64,2}, data));
end;

function outputsFromDataframe( dataframe::DataFrame )
    return Array{Bool,2}(dataframe[:,3]');
end;

dataFromCSV( name::String ) = dataFromDataframe( dataframeFromCSV( name ) );
inputsFromCSV( name::String ) = inputsFromDataframe( dataframeFromCSV( name ) );
outputsFromCSV( name::String ) = outputsFromDataframe( dataframeFromCSV( name ) );