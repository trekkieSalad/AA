using Random

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


function holdOut(N::Int, P::Float64)
    indexes = randperm(N);
    training = Int(round(N*(1-P)));
    return (indexes[1:training], indexes[training+1:end]);
end;

function holdOut(N::Int, validation::Float64, test::Float64)

    (trainValIndex, testIndex) = holdOut(N, test);
    (trainIndex, valIndex) = holdOut(length(trainValIndex), validation*N/length(trainValIndex))

    return (trainValIndex[trainIndex], trainValIndex[valIndex], testIndex);
end;