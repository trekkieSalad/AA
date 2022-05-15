import Pkg;
Pkg.add("Flux")
Pkg.add("Statistics")
Pkg.add("Match")
Pkg.add("Random")
Pkg.add("ProgressMeter")
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("JLD2")
Pkg.add("Images")
Pkg.add("Plots")
Pkg.add("TerminalMenus")
Pkg.add("ScikitLearn")
Pkg.add("ImageMorphology");

using Flux
using Flux.Losses
using Statistics
using Match
using Random
using ProgressMeter
using DataFrames
using CSV
using JLD2
using Images
using Plots
using TerminalMenus
using ScikitLearn

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier