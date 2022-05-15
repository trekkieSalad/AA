
function getSVC(kernel, C, degree, gamma)
    return SVC(kernel=kernel, C=C, degree=degree, gamma=gamma);
end;

function getNeig(n)
    return KNeighborsClassifier(n);
end;

function getTree(n)
    return DecisionTreeClassifier(max_depth=n, random_state=1)
end;

function trainOther(object, inputs, targets, numFolds, met)

    crossValidationIndices = crossvalidation(size(inputs,1), numFolds);
    metrics = Array{Float64,2}(undef,numFolds,length(met));
    mymetrics = Array{Float64,2}(undef,50,length(met));

    for numFold in 1:numFolds

        trainInputs    = inputs[crossValidationIndices.!=numFold, :];
        testInputs     = inputs[crossValidationIndices.==numFold, :];
        trainTargets   = targets[crossValidationIndices.!=numFold];
        testTargets    = targets[crossValidationIndices.==numFold];

        for i in 1:50
            if object[1] == "svm"
                model = getSVC(object[2], object[3], object[4], object[5]);
            elseif object[1] == "knn"
                model = getNeig(object[2]);
            elseif object[1] == "tree"
                model = getTree(object[2]);
            end;
            fit!(model, trainInputs, trainTargets);
            outputs = predict(model, testInputs);
            tmpmetrics = getMetrics(Array{Bool,2}(outputs'), Array{Bool,2}(testTargets'), met);
            mymetrics[i,:] = tmpmetrics;
        end;
        
        metrics[numFold,:] = mean(mymetrics, dims=1);
    end;
    return mean(metrics, dims=1);
end;

function trainSVC(inputs, targets, numFolds, metrics)
    topo = [];
    finals = [];
    kernels = ["linear", "rbf", "sigmoid", "poly"];
    for ker in kernels
        for C in [0.1, 1, 10, 100, 1000, 10000]
            if !(ker == "linear" && C == 1000)
                model = ["svm", ker, C, 3., 2];
                results = trainOther(model, inputs, targets, numFolds, metrics);
                push!(topo, [ker, C]);
                push!(finals, results);
            end;
        end;
    end;
    return topo, finals;
end;

function trainTree(inputs, targets, numFolds, metrics)
    topo = [];
    finals = [];
    for i in 1:20
        model = ["tree", i];
        results = trainOther(model, inputs, targets, numFolds, metrics);
        push!(topo, [i]);
        push!(finals, results);
    end;
    return topo, finals;
end;

function trainNeig(inputs, targets, numFolds, metrics)
    topo = [];
    finals = [];
    for i in 1:20
        model = KNeighborsClassifier(i);  
        model = ["knn", i];  
        results = trainOther(model, inputs, targets, numFolds, metrics);
        push!(topo, [i]);
        push!(finals, results);
    end;
    return topo, finals;
end;