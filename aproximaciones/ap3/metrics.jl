getVP(outputs::Array{Bool,2}, targets::Array{Bool,2}) = count(i->(i==1), outputs .* targets);
getVN(outputs::Array{Bool,2}, targets::Array{Bool,2}) = count(i->(i==0), outputs .+ targets);
getFP(outputs::Array{Bool,2}, targets::Array{Bool,2}) = count(i->(i==1), outputs .- targets);
getFN(outputs::Array{Bool,2}, targets::Array{Bool,2}) = count(i->(i==-1), outputs .- targets);

accuracy(outputs::Array{Bool,2}, targets::Array{Bool,2}) = mean(outputs.==targets);
errorRate(outputs::Array{Bool,2}, targets::Array{Bool,2}) = mean(outputs.!=targets);
recall(outputs::Array{Bool,2}, targets::Array{Bool,2}) = getVP(outputs,targets)/(getVP(outputs,targets)+getFN(outputs,targets));
specificity(outputs::Array{Bool,2}, targets::Array{Bool,2}) = getVN(outputs,targets)/(getVN(outputs,targets)+getFP(outputs,targets));
ppv(outputs::Array{Bool,2}, targets::Array{Bool,2}) = getVP(outputs,targets)/(getVP(outputs,targets)+getFP(outputs,targets));
npv(outputs::Array{Bool,2}, targets::Array{Bool,2}) = getVN(outputs,targets)/(getVN(outputs,targets)+getFN(outputs,targets));
f1(outputs::Array{Bool,2}, targets::Array{Bool,2}) = 2*((recall(outputs,targets)*ppv(outputs,targets))/(recall(outputs,targets)+ppv(outputs,targets)));

function showMetrics(outputs::Array{Bool,2}, targets::Array{Bool,2}, metrics::Array{String,1})
    for i in 1:length(metrics)
        val = metrics[i];
        @match val begin
            "VP"            => println(val, ": ", getVP(outputs,targets));
            "VN"            => println(val, ": ", getVN(outputs,targets));
            "FP"            => println(val, ": ", getFP(outputs,targets));
            "FN"            => println(val, ": ", getFN(outputs,targets));
            "Accuracy"      => println(val, ": ", accuracy(outputs,targets));
            "Error Rate"    => println(val, ": ", errorRate(outputs,targets));
            "Recall"        => println(val, ": ", recall(outputs,targets));
            "Specificity"   => println(val, ": ", specificity(outputs,targets));
            "PPV"           => println(val, ": ", ppv(outputs,targets));
            "NPV"           => println(val, ": ", npv(outputs,targets));
            "F1"            => println(val, ": ", f1(outputs,targets));
            _               => println("Unknown metric: ", val); 
        end
    end;
end;

showMetrics(outputs::Array{Bool,2}, targets::Array{Bool,2}) = showMetrics(outputs,targets,["VP", "VN", "FP", "FN", "Accuracy", "Error Rate", "Recall", "Specificity", "PPV", "NPV", "F1"]);

function getMetrics(outputs::Array{Bool,2}, targets::Array{Bool,2}, metrics::Array{String,1})
    results = [];
    for i in 1:length(metrics)
        val = metrics[i];
        @match val begin
            "VP"            => push!(results, getVP(outputs,targets));
            "VN"            => push!(results, getVN(outputs,targets));
            "FP"            => push!(results, getFP(outputs,targets));
            "FN"            => push!(results, getFN(outputs,targets));
            "Accuracy"      => push!(results, accuracy(outputs,targets));
            "Error Rate"    => push!(results, errorRate(outputs,targets));
            "Recall"        => push!(results, recall(outputs,targets));
            "Specificity"   => push!(results, specificity(outputs,targets));
            "PPV"           => push!(results, ppv(outputs,targets));
            "NPV"           => push!(results, npv(outputs,targets));
            "F1"            => push!(results, f1(outputs,targets));
            _               => println("Unknown metric: ", val); 
        end
    end;
    return results;
end;

getMetrics(outputs::Array{Bool,2}, targets::Array{Bool,2}) = getMetrics(outputs,targets,["VP", "VN", "FP", "FN", "Accuracy", "Error Rate", "Recall", "Specificity", "PPV", "NPV", "F1"]);
