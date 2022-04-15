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

ONV(opt::String) = println("La opcion introducida '",opt,"' no es valida")

###############################################################################################

function cabecera()
    Base.run(`clear`);
    println("+-------------------------------------------------+");
    println("|                                                 |");
    println("|           Entrenador de RR.NN.AA                |");
    println("|                                                 |");
    println("+-------------------------------------------------+\n");
end;

function setTopologies()
    println("Introduzca linea a linea las topologias que desea probar: ");
    println("(para terminar introduzca una linea en blanco)");
    topologies = subdivideTopology(readModels());
    return topologies;
end;

function setMetrics()

    mensaje = "seleccione las metricas a utilizar en la evaluacion: "
    options = [ "VP", "VN", "FP", "FN", "Precision", "Ratio de error", "Sensibilidad", "Especificidad", "Valor predictivo positivo", 
                "Valor predictivo negativo", "F1-Score" ]
    menu = MultiSelectMenu(options, pagesize = 5);
    selection = request(mensaje, menu);
    metrics = String[];
    for el in selection
        push!(metrics, options[el])
    end
    return metrics;
end;

function setIters(n::String)
    print(n);
    iter = parse(Int64, readline());
    return iter;
end;


###############################################################################################

function execute()

    cabecera();

    print("Desea crear crear un nuevo DataSet? y|n : ");
    nd = readline();

    if nd == "y"
        dataset = createDataSet();
    
    elseif nd == "n"
        print("\nIntroduzca el nombre de un fichero CSV del que leer el dataset: ")
        datasetName = readline();
        dataset = dataframeFromCSV( datasetName );

    else
        ONV(nd);
        
    end;

    #inputs = inputsFromDataframe(dataset);
    #outputs = outputsFromDataframe(dataset);

    cabecera();

    options = [ "Topologia", "Metricas", "Nº de iteraciones (entrenamientos para una topologia) <50>", "Nº de subconjuntos (CrossValidation) <10>", 
                "Ciclos sin mejorar validacion <100>", "Ratio de aprendizaje <0.01>", "Ciclos maximos de entrenamiento <5000>"];

    menu = MultiSelectMenu(options, pagesize = 5);

    selection = request("Selecciona los campos que deseas modificar \n(los campos no seleccionados tomarán valores por defecto indicados):", menu);

    topologies = [[-1]]
    metrics = [ "VP", "VN", "FP", "FN", "Precision", "Ratio de error", "Sensibilidad", "Especificidad", "Valor predictivo positivo", 
                "Valor predictivo negativo", "F1-Score" ]
    iter = 50;
    folds = 10;
    early = 100;
    ratio = 0.01;
    maxCycles = 5000;

    for value in selection
        cabecera();
        @match value begin
            1   => (topologies = setTopologies());
            2   => (metrics = setMetrics());
            3   => (iter = setIters("Introduzca el numero de iteraciones en cada fold: "));
            4   => (folds = setIters("Introduzca el numero de subconjuntos para el entrenamiento: "));
            5   => return
            6   => return
            7   => return
            _   => return
        end;
    end;

    println(topologies);
    println(metrics);
    println(iter);
    println(folds);


end;
