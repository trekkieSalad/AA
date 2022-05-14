
using Flux
using Flux.Losses
using Flux: onehotbatch, onecold
using JLD2, FileIO
using Statistics: mean

function loadFolderImages(folderName::String)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName));
            image = imresize(image, (50, 50));
            image = convert(Array{Float32, 2}, gray.(Gray.(image)));
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
targets = [trues(length(positives)); falses(length(negatives))];
(trainInd, testInd) = holdOut(106, 0.2);

train_imgs = inputs[trainInd];
test_imgs = inputs[testInd];
train_targets = targets[trainInd]';
test_targets = targets[testInd]';


function convertirArrayImagenesHWCN(imagenes)
    numPatrones = length(imagenes);
    #println(length(imagenes));
    nuevoArray = Array{Float32,4}(undef, 50, 50, 1, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (size(imagenes[i])==(50,50)) "Las imagenes no tienen tamaño 28x28";
        nuevoArray[:,:,1,i] .= imagenes[i][:,:];
    end;
    return nuevoArray;
end;
train_imgs = convertirArrayImagenesHWCN(train_imgs);
test_imgs = convertirArrayImagenesHWCN(test_imgs);

println("Tamaño de la matriz de entrenamiento: ", size(train_imgs), " ", size(train_targets));
println("Tamaño de la matriz de test:          ", size(test_imgs), " ", size(test_targets));


# Cuidado: en esta base de datos las imagenes ya estan con valores entre 0 y 1
# En otro caso, habria que normalizarlas
println("Valores minimo y maximo de las entradas: (", minimum(train_imgs), ", ", maximum(train_imgs), ")");

train_set = (train_imgs, train_targets);
test_set = (test_imgs, test_targets);


# Hago esto simplemente para liberar memoria, las variables train_imgs y test_imgs ocupan mucho y ya no las vamos a usar
train_imgs = nothing;
test_imgs = nothing;
GC.gc(); # Pasar el recolector de basura




funcionTransferenciaCapasConvolucionales = relu;

# Definimos la red con la funcion Chain, que concatena distintas capas
ann = Chain(

    Conv((3, 3), 1=>8, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 8=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(1152, 1, σ),
    softmax
)



numImagenEnEseBatch = [1,85];
entradaCapa = train_set[1][:,:,:,numImagenEnEseBatch];
numCapas = length(params(ann));
println("La RNA tiene ", numCapas, " capas:");
for numCapa in 1:numCapas
    println("   Capa ", numCapa, ": ", ann[numCapa]);
    # Le pasamos la entrada a esta capa
    global entradaCapa # Esta linea es necesaria porque la variable entradaCapa es global y se modifica en este bucle
    capa = ann[numCapa];
    salidaCapa = capa(entradaCapa);
    println("      La salida de esta capa tiene dimension ", size(salidaCapa));
    entradaCapa = salidaCapa;
end

ann(train_set[1][:,:,:,numImagenEnEseBatch]);

loss(x, y) = Losses.binarycrossentropy(ann(x),y);;

accuracy(batch) = mean(onecold(ann(batch[1])) .== batch[2]);

println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*mean(accuracy(train_set)), " %");


# Optimizador que se usa: ADAM, con esta tasa de aprendizaje:
opt = ADAM(0.01);


println("Comenzando entrenamiento...")
mejorPrecision = -Inf;
criterioFin = false;
numCiclo = 0;
numCicloUltimaMejora = 0;
mejorModelo = nothing;

while (!criterioFin)

    global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin;

    # Se entrena un ciclo
    Flux.train!(loss, params(ann), [(train_set[1], train_set[2])], opt);

    numCiclo += 1;

    # Se calcula la precision en el conjunto de entrenamiento:
    precisionEntrenamiento = mean(accuracy(train_set));
    println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

    # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
    if (precisionEntrenamiento > mejorPrecision)
        mejorPrecision = precisionEntrenamiento;
        precisionTest = accuracy(test_set);
        println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
        mejorModelo = deepcopy(ann);
        numCicloUltimaMejora = numCiclo;
    end

    # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
    if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
        opt.eta /= 10.0
        println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta);
        numCicloUltimaMejora = numCiclo;
    end

    # Criterios de parada:

    # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
    if (precisionEntrenamiento >= 0.999)
        println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
        criterioFin = true;
    end

    # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
    if (numCiclo - numCicloUltimaMejora >= 10)
        println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
        criterioFin = true;
    end
end