function  plotMeanStd( data::DataFrame )
    grupos = groupby(data, :Column3);
    tumors = grupos[(Column3 = true,)];
    no_tumors = grupos[(Column3 = false,)];

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
    
    Plots.scatter(xt,yt, color="red", label="Tumores")
    Plots.scatter!(xn,yn, color="blue", label="No Tumores")
end;

function plotLossTrain( losses::Tuple )
    Plots.plot(0:length(losses[1])-1, losses[1], color="red", label="Training")
    Plots.plot!(0:length(losses[1])-1, losses[2], color="blue", label="Test")
    Plots.plot!(0:length(losses[1])-1, losses[3], color="green", label="Validation")
end;