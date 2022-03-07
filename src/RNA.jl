using DataFrames
using CSV

data = DataFrame(CSV.File("tumores.csv", header=false));
rna_inputs = [data[!,1] data[!,2]];
targets = data[!,3];

