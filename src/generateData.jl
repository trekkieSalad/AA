function imageToColorArray(image::Array)
    image2 = withInversion(image);
    matrix = Array{Float64,2}(undef,1,4)
    matrix[1] = mean(convert(Array{Float64,2}, gray.(Gray.(image))));
    matrix[2] = std(convert(Array{Float64,2}, gray.(Gray.(image))));

    matrix[3] = mean(convert(Array{Float64,2}, gray.(Gray.(image2))));
    matrix[4] = std(convert(Array{Float64,2}, gray.(Gray.(image2))));

    return matrix;
end;
#imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

function withInversion(image::Array{RGB{Normed{UInt8,8}},2})
    # ha, va
    image=convert(Array{Float64,2}, gray.(Gray.(image)))
    # separa los elementos de la imagen mayores que 0.5 y menores que 0.5 en 2 matrices

    mayores = image .> 0.5;
    menores = image .< 0.5;

    horizontalDiference = broadcast(abs, (image .- reverse(image, dims=2)));
    verticalDiference = broadcast(abs, (image .- reverse(image, dims=1)));
    horizontalAddition = broadcast(abs, (image .+ (reverse(image, dims=2) .* mayores) .- (reverse(image, dims=2) .* menores)));
    verticalAddition = broadcast(abs, (image .+ reverse(image, dims=1)));
    horizontalTotalAdd = broadcast(abs, (image .+ (image .* mayores) .- (1 .- image .* menores)));
    verticalTotalAdd = broadcast(abs, (image .+ (reverse(image, dims=2) .* mayores) .- (1 .- reverse(image, dims=2) .* menores)));

    return verticalDiference;
end;

function loadFolderImages(folderName::String, complex::Bool=false)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName));
            if complex
                image = withInversion(image);
            end;
            #image = withInversion(image);
            #@assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
            push!(images, image);
        end;
    end;
    return imageToColorArray.(images);
end;

function loadTrainingDataset(pos::String, neg::String, complex::Bool=false)
    df = DataFrame(mean = Float64[], std = Float64[], meanadd = Float64[], stdadd = Float64[]);
    #df = DataFrame(mean = Float64[], std = Float64[]);

    positives = loadFolderImages(pos, complex);
    negatives = loadFolderImages(neg, complex);
    inputs = [positives; negatives];

    targets = [trues(length(positives)); falses(length(negatives))];

    for i = 1:length(inputs)
        push!(df, (inputs[i][1], inputs[i][2], inputs[i][3], inputs[i][4]));
        #push!(df, (inputs[i][1], inputs[i][2]));
    end;

    df.target = targets;

    return df;
end;

function createDataSet(complex::Bool = false, name::String = "tumores", onlyCreate::Bool = false)
    dir = pwd();
    pos = dir * "/brain_tumor_classification/tumor";
    neg = dir * "/brain_tumor_classification/no_tumor";
    salida = loadTrainingDataset(pos,neg, complex);
    CSV.write(name * ".csv", salida, header=false);
    if !onlyCreate
        return salida;
    end;
end;