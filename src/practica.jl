using JLD2
using Images
using Statistics

function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float64,2}(undef,1,2)
    matrix[1] = mean(convert(Array{Float64,2}, gray.(Gray.(image))));
    matrix[2] = std(convert(Array{Float64,2}, gray.(Gray.(image))));
    return matrix;
end;
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

function loadFolderImages(folderName::String)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName));
            @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
            push!(images, image);
        end;
    end;
    return imageToColorArray.(images);
end;

function loadTrainingDataset(pos::String, neg::String)
    positives = loadFolderImages(pos);
    negatives = loadFolderImages(neg);
    targets = [trues(length(positives)); falses(length(negatives))];
    return ([positives; negatives], targets);
end;

pos = "/home/fuhrerg/GIT/AA/brain_tumor_classification/tumor";
neg = "/home/fuhrerg/GIT/AA/brain_tumor_classification/no_tumor";
salida = loadTrainingDataset(pos,neg);