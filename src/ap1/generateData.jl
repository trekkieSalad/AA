function imageToColorArray(image::Array)
    #image2 = withInversion(image);
    matrix = Array{Float64,2}(undef,1,2)
    matrix[1] = mean(convert(Array{Float64,2}, gray.(Gray.(image))));
    matrix[2] = std(convert(Array{Float64,2}, gray.(Gray.(image))));

    return matrix;
end;


function loadFolderImages(folderName::String)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName));
            push!(images, image);
        end;
    end;
    return imageToColorArray.(images);
end;

function loadTrainingDataset(pos::String, neg::String)
    df = DataFrame(mean = Float64[], std = Float64[]);

    positives = loadFolderImages(pos);
    negatives = loadFolderImages(neg);
    inputs = [positives; negatives];

    targets = [trues(length(positives)); falses(length(negatives))];

    for i = 1:length(inputs)
        push!(df, (inputs[i][1], inputs[i][2]));
    end;

    df.target = targets;

    return df;
end;

function createDataSet(name::String = "tumores", onlyCreate::Bool = false)
    dir = pwd();
    pos = dir * "/brain_tumor_classification/tumor";
    neg = dir * "/brain_tumor_classification/no_tumor";
    salida = loadTrainingDataset(pos,neg);
    CSV.write(name * ".csv", salida, header=false);
    if !onlyCreate
        return salida;
    end;
end;