#--------------------------------------------------------------------------
#                         METRICAS
#--------------------------------------------------------------------------

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert(length(outputs)==length(targets));

    confusion = Array{Int64,2}(undef, 2, 2);

    confusion[1,1] = sum(.!targets .& .!outputs); # VN
    confusion[1,2] = sum(.!targets .&   outputs); # FP

    confusion[2,1] = sum(  targets .& .!outputs); # FN
    confusion[2,2] = sum(  targets .&   outputs); # VP
    
    acc         = accuracy(outputs, targets); # definida anteriormente
    errorRate   = 1. - acc;
    recall      = mean(  outputs[  targets]);
    specificity = mean(.!outputs[.!targets]);
    VPP         = mean(  targets[  outputs]);
    VPN         = mean(.!targets[.!outputs]);

    if isnan(recall) && isnan(VPP)
        recall = 1.;
        VPP = 1.;
    elseif isnan(specificity) && isnan(VPN) 
        specificity = 1.;
        VPN = 1.;
    end;
    
    recall      = isnan(recall)      ? 0. : recall;
    specificity = isnan(specificity) ? 0. : specificity;
    VPP         = isnan(VPP)   ? 0. : VPP;
    VPN         = isnan(VPN)         ? 0. : VPN;
    F1          = (recall==VPP==0.) ? 0. : 2*(recall*VPP)/(recall+VPP);

    return (acc, errorRate, recall, specificity, VPP, VPN, F1, confusion)
end;

confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = confusionMatrix(Array{Bool,1}(outputs.>=threshold), targets);

#--------------------------------------------------------------------------
#                         ONEVSALL
#--------------------------------------------------------------------------

function oneVSall(inputs::Array{Float64,2}, targets::AbstractArray{Bool,2})
    numeroClases = size(targets,2);
    @assert(numeroClases>2);
    
    outputs = Array{Float64,2}(undef, size(inputs,1), numeroClases);
    for numClass in 1:numeroClases
        model = fit(inputs, targets[:,[numClass]]);
        outputs[:,numClass] .= model(inputs);
    end;
    
    outputs = collect(softmax(outputs')');
    outputs = classifyOutputs(outputs);
    classComparison = (targets .== outputs);
    correctClassifications = all(classComparison, dims=2);
    return mean(correctClassifications);
end;

#--------------------------------------------------------------------------
#                         METRICASMULTICLASE
#--------------------------------------------------------------------------

function confusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    numeroClases = size(targets,2);
    @assert(numeroClases!=2);

    if (numeroClases==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    else
        @assert(all(sum(outputs, dims=2).==1));
        
        recall      = zeros(numeroClases);
        specificity = zeros(numeroClases);
        PPV         = zeros(numeroClases);
        NPV         = zeros(numeroClases);
        F1          = zeros(numeroClases);
        
        confMatrix  = Array{Int64,2}(undef, numeroClases, numeroClases);
        instanciasPorClase = vec(sum(targets, dims=1));
        for numClass in findall(instanciasPorClase.>0)
            (_, _, recall[numClass], specificity[numClass], PPV[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
        end;

        confMatrix = Array{Int64,2}(undef, numeroClases, numeroClases);
        for claseObjetivo in 1:numeroClases, claseSalida in 1:numeroClases
            confMatrix[claseObjetivo, claseSalida] = sum(targets[:,claseObjetivo] .& outputs[:,claseSalida]);
        end;

        if weighted
            pesos = instanciasPorClase./sum(instanciasPorClase);
            recall      = sum(pesos.*recall);
            specificity = sum(pesos.*specificity);
            PPV         = sum(pesos.*PPV);
            NPV         = sum(pesos.*NPV);
            F1          = sum(pesos.*F1);
        else #macro
            clasesInstanciadas = sum(instanciasPorClase.>0);
            recall      = sum(recall)/clasesInstanciadas;
            specificity = sum(specificity)/clasesInstanciadas;
            PPV         = sum(PPV)/clasesInstanciadas;
            NPV         = sum(NPV)/clasesInstanciadas;
            F1          = sum(F1)/clasesInstanciadas;
        end;
        
        acc = accuracy(outputs, targets);
        errorRate = 1 - acc;

        return (acc, errorRate, recall, specificity, PPV, NPV, F1, confMatrix);
    end;
end;


function confusionMatrix(outputs::AbstractArray{<:Any}, targets::AbstractArray{<:Any}; weighted::Bool=true)
    @assert(all([in(output, unique(targets)) for output in outputs]));
    classes = unique(targets);
    return confusionMatrix(oneHotEncoding(outputs), oneHotEncoding(targets); weighted=weighted);
end;

confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);





#--------------------------------------------------------------------------
#                         PRINTEARRESULTADOS
#--------------------------------------------------------------------------



function printConfusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, PPV, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numeroClases = size(confMatrix,1);
    aciertos = 0;
    fallos = 0;
    for claseObjetivo in 1:numeroClases
        for compar in 1:numeroClases
            if claseObjetivo == compar
                aciertos += confMatrix[claseObjetivo,compar]
            else
                fallos += confMatrix[claseObjetivo,compar]
            end;
        end;
    end;
    println("Aciertos: ", aciertos);
    println("Fallos: ", fallos);
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("PPV: ", PPV);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    println("-----------------------------")
    return (acc, errorRate, recall, specificity, PPV, NPV, F1, confMatrix);
end;



topologia = [3, 2];
ratio = 0.01;
ciclos = 1000;
validar = 0.2;
testear = 0.2;
ciclosValidacion = 6;


dataset = readdlm("fonts/iris.data",',');
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);

numeroClases = size(targets,2);
@assert(numeroClases>2);
normalizeMinMax!(inputs);

(trainingIndices, validationIndices, testIndices) = holdOut(size(inputs,1), validar, testear);

trainInputs    = inputs[trainingIndices,:];
valInputs  = inputs[validationIndices,:];
testInputs        = inputs[testIndices,:];
trainTargets   = targets[trainingIndices,:];
valTargets = targets[validationIndices,:];
testTargets       = targets[testIndices,:];

outputs = Array{Float64,2}(undef, size(inputs,1), numeroClases);

for numClass = 1:numeroClases

    local ann;
    ann, = trainClassANN(topologia,
        (trainInputs,   trainTargets[:,[numClass]]),
        (valInputs, valTargets[:,[numClass]]),
        (testInputs,       testTargets[:,[numClass]]),
        maxEpochs=ciclos, learningRate=ratio, maxEpochsVal=ciclosValidacion);

    outputs[:,numClass] = ann(inputs')';

end;

outputs = collect(softmax(outputs')');

println("\nENTRENAMIENTO:")
printConfusionMatrix(outputs[trainingIndices,:], trainTargets; weighted=true);
println("\nVALIDACION:")
printConfusionMatrix(outputs[validationIndices,:], valTargets; weighted=true);
println("\nTEST:")
printConfusionMatrix(outputs[testIndices,:], testTargets; weighted=true);
