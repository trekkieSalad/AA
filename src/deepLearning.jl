using Flux
using Flux.Losses
using Images


function convertirArrayImagenesHWCN(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, 28, 28, 1, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (size(imagenes[i])==(28,28)) "Las imagenes no tienen tamaño 28x28";
        nuevoArray[:,:,:,i] .= imagenes[i][:,:,:];
    end;
    return nuevoArray;
end

imageToGrayArray(image:: Array{RGB{Normed{UInt8,8}},2}) = convert(Array{Float64,2}, gray.(Gray.(image)));

function redesConvolucionales(images, size = (28,28))
    resize_img(img) = imresize(img, size)
    images = resize_img.(images)
    images = imageToGrayArray.(images)
    images = convertirArrayImagenesHWCN(images)
end

function crearRNAConvolucional(topology :: AbstractArray{Tuple{Int64, Int64}}, fPooling = MaxPool, fTransferencia = relu)
    if (length(topology) > 1)
        ann = Chain()

        for (iConv, oConv) in topology[1:(length(topology)-1)]
            ann = Chain(ann..., Conv((3, 3), iConv=>oConv, pad=(1,1), fTransferencia));
            ann = Chain(ann..., fPooling((2,2)));
        end

        ann = Chain(ann..., x -> reshape(x, :, size(x, 4)));

        (iDense, oDense) = topology[length(topology)];
        if (oDense >= 2)
            ann = Chain(ann..., Dense(iDense, oDense), softmax)
        else
            ann = Chain(ann..., Dense(iDense, oDense, σ))
        end

        return ann;
    else
        throw(ErrorException("topology no tiene el tamaño suficiente"));
    end
end

# Hay que declarar las variables globales que van a ser modificadas en el interior del bucle
global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin;

function entrenarRNAConvolucional(ann, dataset::Tuple{Array{Float32,1}, Array{Bool,2}},
    test::Tuple{Array{Float32,1}, Array{Bool,2}}, learningRate :: Real = 0.01, minPrecision :: Real = 0.999)

    opt = ADAM(learningRate);
    #loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    loss(x, y) = Losses.binarycrossentropy(ann(x),y);


    println("Comenzando entrenamiento...")
    mejorPrecision = -Inf;
    criterioFin = false;
    numCiclo = 0;
    numCicloUltimaMejora = 0;
    mejorModelo = nothing;
    results = Array{Float32,1}(undef, 0)

    while (!criterioFin)

        # Se entrena un ciclo
        #Flux.train!(loss, params(ann), [(inputsTrain, oneHotEncoding(targetsTrain)')], opt);
        Flux.train!(loss, params(ann), [(dataset[1], dataset[2])], opt);

        numCiclo += 1;

        #outAnn = ann(dataset[1]).>=0.5;

        # Se calcula la precision en el conjunto de entrenamiento:
        precisionEntrenamiento = accuracy(dataset[2], ann(dataset[1])', 0.5);
        push!(results, precisionEntrenamiento);
        println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

        # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
        if (precisionEntrenamiento >= mejorPrecision)
            #outTest = ann(test[1]).>=0.5;
            mejorPrecision = precisionEntrenamiento;
            precisionTest = accuracy(test[2], ann(test[1])', 0.5);
            println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
            mejorModelo = deepcopy(ann);
            numCicloUltimaMejora = numCiclo;
        end

        # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
        if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
            opt.eta /= 10.0;
            println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta);
            numCicloUltimaMejora = numCiclo;
        end

        # Criterios de parada:

        # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
        if (precisionEntrenamiento >= minPrecision)
            println("   Se para el entrenamiento por haber llegado a una precision de ", 100*minPrecision ," %")
            criterioFin = true;
        end

        # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
        if (numCiclo - numCicloUltimaMejora >= 10)
            println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
            criterioFin = true;
        end
    end

    return (mejorModelo, results)
end




function loadFolderImages(folderName::String)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName));
            push!(images, image);
        end;
    end;
    return images;
end;

dir = pwd();
pos = dir * "/brain_tumor_classification/tumor";
neg = dir * "/brain_tumor_classification/no_tumor";

positives = loadFolderImages(pos);
negatives = loadFolderImages(neg);
inputs = [positives; negatives];
inputs = redesConvolucionales(inputs);
targets = [trues(length(positives)); falses(length(negatives))];
(trainInd, testInd) = holdOut(106, 0.2);

train_imgs = inputs[trainInd];
test_imgs = inputs[testInd];
train_targets = targets[trainInd]';
test_targets = targets[testInd]';
println(size(train_targets));
println(size(test_targets));
train_targets = convert(Array{Bool,2}, train_targets);
test_targets = convert(Array{Bool,2}, test_targets);
ann = crearRNAConvolucional([(1,8);(8,16);(16,32)])

entrenarRNAConvolucional(ann, (train_imgs, train_targets),(test_imgs,test_targets))