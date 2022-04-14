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
    while line != ""
        line = readline();
        if line != ""
            push!(result, line)
        end;
    end;
    return result;
end;

###############################################################################################

function execute()
    println("+-------------------------------------------------+");
    println("|                                                 |");
    println("|           Entrenador de RR.NN.AA                |");
    println("|                                                 |");
    println("+-------------------------------------------------+");

end;
