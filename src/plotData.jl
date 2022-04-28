function  plotMeanStd( data::DataFrame )
    grupos = groupby(data, :target);
    tumors = grupos[(target = true,)];
    no_tumors = grupos[(target = false,)];

    xt = tumors[:,1]
    yt = tumors[:,2]
    xn = no_tumors[:,1]
    yn = no_tumors[:,2]
    
    Plots.scatter(xt,yt, color="red", label="Tumores")
    Plots.scatter!(xn,yn, color="blue", label="No Tumores")
end;

function  plotMeanStd( data::Array{Float64,2} )
    mid = convert(Int64,size(data,2) / 2);
    print(1:mid)
    print((mid+1):size(data,2))
    data = convert(Array{Float64,2}, data');
    tumors = data[1:53, :];
    no_tumors = data[54:106, :];

    data

    xt = tumors[:,1]
    yt = tumors[:,2]
    xn = no_tumors[:,1]
    yn = no_tumors[:,2]
    
    Plots.scatter(xt,yt, color="red", label="Tumores", legend=:bottomright,  xlabel="mean", ylabel="std")
    Plots.scatter!(xn,yn, color="blue", label="No Tumores")
end;

function plotLossTrain( losses::Tuple )
    Plots.plot(0:length(losses[1])-1, losses[1], color="red", label="Training")
    Plots.plot!(0:length(losses[1])-1, losses[2], color="blue", label="Test")
    Plots.plot!(0:length(losses[1])-1, losses[3], color="green", label="Validation")
end;

function plotBests(name1, name2)
    recall = [];
    brecall = [];
    specificity = [];
    bspecificity = [];
    open(name1) do f       
        # read till end of file
        while ! eof(f) 
       
            # read a new / next line for every iteration          
            s = readline(f)   
            if startswith(s, "\tRecall:") || startswith(s, "\tSpecificity:")
                s = split(s, ":");
                val = parse(Float64, s[2]);
                if startswith(s[1], "\tRecall")
                    push!(recall, val);
                else
                    push!(specificity, val);
                end;
            end

        end       
    end
    open(name2) do f       
        # read till end of file
        while ! eof(f) 
       
            # read a new / next line for every iteration          
            s = readline(f)   
            if startswith(s, "\tRecall:") || startswith(s, "\tSpecificity:")
                s = split(s, ":");
                val = parse(Float64, s[2]);
                if startswith(s[1], "\tRecall")
                    push!(brecall, val);
                else
                    push!(bspecificity, val);
                end;
            end

        end       
    end

    Plots.scatter(specificity,recall, color="red", xlabel="Specificity", ylabel="Recall", label="all topologies", legend=:bottomleft, xticks = :all)
    Plots.scatter!(bspecificity,brecall, color="blue", label="best recall topologies")
end;

function plotNBests(n::Int64, name::String, type::String)
    r, s, ir, is, ib = getNBestsFromFile(n, name);
    Plots.scatter(s,r, color="red", xlabel="Specificity", ylabel="Recall", label="all "*type, legend=:bottomleft, xticks = :all)
    Plots.scatter!(s[ir],r[ir], color="blue", label="bests recall "*type)
    Plots.scatter!(s[is],r[is], color="green", label="bests specificity "*type)
    Plots.scatter!(s[ib],r[ib], color="yellow", label="bests recall and specificity "*type, markershape = :square)
end;