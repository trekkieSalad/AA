function subdivide(n::String, type)        
    array = split(join(map(x -> isspace(n[x]) ? "" : n[x], 1:length(n))),",");
    topology = Array{type, 1}(undef, length(array));
    for i in 1:length(array)
        if type == String
            topology[i] = array[i]
        else
            topology[i] = parse(type, array[i]);
        end;
    end
    return topology;
end;

subdivideParameter(n::String) = subdivide(n, String);

function subdivideTopology(n::Array{String, 1})
    topologies = Array{Int64,1}[];
    for el in 1:length(n)
        topology = subdivide(n[el], Int64);
        push!(topologies, topology);
    end;
    return topologies;
end;

function readModels()
    result = String[];
    line = "init";
    while line != "" || !(length(result) > 0)
        line = readline();
        if line != ""
            push!(result, line)
        end;
    end;
    return result;
end;

ONV(opt::String) = println("La opcion introducida '",opt,"' no es valida\n")

###############################################################################################

function cabecera()
    Base.run(`clear`);
    println("\t+-------------------------------------------------+");
    println("\t|                                                 |");
    println("\t|           Entrenador de RR.NN.AA                |");
    println("\t|                                                 |");
    println("\t+-------------------------------------------------+\n");
end;

function setTopologies()
    println("Introduzca linea a linea las topologias que desea probar: ");
    println("(para terminar introduzca una linea en blanco)\n");
    topologies = subdivideTopology(readModels());
    return topologies;
end;

function setMetrics(finals, m)

    
    menu = MultiSelectMenu(finals, pagesize = 5);
    selection = request(m, menu);
    sorted = sort!(collect(selection));
    metrics = String[];
    for el in sorted
        push!(metrics, finals[el])
    end
    return metrics;
end;

setMetrics() = setMetrics(["VP", "VN", "FP", "FN", "Accuracy", "Error Rate", "Recall", "Specificity", "PPV", "NPV", "F1"], "Seleccione las metricas a mostrar: ")

function setIters(n::String, type)
    print(n);
    iter = parse(type, readline());
    return iter;
end;


###############################################################################################

function execute()

    @label inicio;
    cabecera();

    if @isdefined(nd)
        ONV(nd);
    end;

    print("Desea crear crear un nuevo DataSet? y|n : ");
    nd = readline();

    if nd == "y"
        dataset = createDataSet();
    
    elseif nd == "n"
        print("\nIntroduzca el nombre de un fichero CSV del que leer el dataset: ")
        datasetName = readline();
        dataset = dataframeFromCSV( datasetName );

    else
        @goto inicio;
    end;

    inputs = inputsFromDataframe(dataset);
    outputs = outputsFromDataframe(dataset);

    otherInputs = otherInputsFromDataframe(dataset);
    otherOutputs = otherOutputsFromDataframe(dataset);

    cabecera();

    options = [ "Topologia", "Metricas <Precision>", "Metricas de evaluacion <Precision>", "Nº de iteraciones (entrenamientos para una topologia) <50>", "Nº de subconjuntos (CrossValidation) <10>", 
                "Ciclos sin mejorar validacion <100>", "Ratio de aprendizaje <0.01>", "Ciclos maximos de entrenamiento <5000>"];

    menu = MultiSelectMenu(options, pagesize = 5);

    selection = request("Selecciona los campos que deseas modificar \n(los campos no seleccionados tomarán valores por defecto indicados):", menu);
    sortedSelection = sort!(collect(selection));

    topologies = [[-1]]
    metrics = ["Accuracy"]
    eMetrics = ["Accuracy"]
    iter = 50;
    folds = 10;
    early = 300;
    ratio = 0.01;
    maxCycles = 7000;

    for value in sortedSelection
        cabecera();
        @match value begin
            1   => (topologies = setTopologies());
            2   => (metrics = setMetrics());
            3   => (eMetrics = setMetrics(metrics, "Seleccione las metricas a utilizar en la evaluacion: "));
            4   => (iter = setIters("Introduzca el numero de iteraciones en cada fold: ", Int64));
            5   => (folds = setIters("Introduzca el numero de subconjuntos para el entrenamiento: ", Int64));
            6   => (early = setIters("Introduzca el numero de ciclos sin mejora antes de parar el entrenamiento: ", Int64));
            7   => (ratio = setIters("Introduzca el ratio de aprendizaje para el entrenamiento: ", Float64));
            8   => (maxCycles = setIters("Introduzca el numero de máximo de ciclos de entrenamiento: ", Int64));
        end;
    end;

    @label nTop

    cabecera();

    print("Desea obtener un top de las mejores N topologias? y|n : ");
    bests = readline();

    if bests != "y" && bests != "n"
        ONV(nd);
        @goto nTop;
    end;

    if bests == "y"
        n = setIters("Introduzca el numero de topologias que quiere en el top: ", Int64);
    end;

    cabecera();

    topo, resultsByFold, results = trainAllRNA(inputs,outputs,metrics,trainIterations=iter, numFolds=folds, maxCycle=maxCycles, earlyStoppingEpochs=early, minLoss=0.0, learningRate=ratio, rnaLayers=topologies)

    cabecera();

    resultsToFile(results, resultsByFold, topo, metrics, "rna.txt");

    if bests == "y"
        finalTopos, finalResultsByFold, finalResults = getNBests(n, topo, resultsByFold, results, eMetrics, metrics);
        resultsToFile(finalResults, finalResultsByFold, finalTopos, metrics, "bestsRna.txt");
    end;

    trees, resultTrees = trainTree(otherInputs, otherOutputs, folds, metrics);
    neigs, resultNeigs = trainNeig(otherInputs, otherOutputs, folds, metrics);
    svcs, resultSVCs = trainSVC(otherInputs, otherOutputs, folds, metrics);

    resultsToFile(resultTrees, [], trees, metrics, "trees.txt");
    resultsToFile(resultNeigs, [], neigs, metrics, "neigs.txt");
    resultsToFile(resultSVCs, [], svcs, metrics, "svms.txt");

    if bests == "y"
        finalTrees, finalResultsByFold, finalResultsTrees = getNBests(n, trees, [], resultTrees, eMetrics, metrics);
        finalNeigs, finalResultsByFold, finalResultsNeigs = getNBests(n, neigs, [], resultNeigs, eMetrics, metrics);
        finalSvcs, finalResultsByFold, finalResultsSvcs = getNBests(n, svcs, [], resultSVCs, eMetrics, metrics);

        resultsToFile(finalResultsTrees, [], finalTrees, metrics, "bestsTrees.txt");
        resultsToFile(finalResultsNeigs, [], finalNeigs, metrics, "bestsNeigs.txt");
        resultsToFile(finalResultsSvcs, [], finalSvcs, metrics, "bestsSvms.txt");
    end;

end;
