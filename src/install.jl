if !@isdefined(install)
    include("dependencies.jl")
end;
include("generateData.jl")
include("metrics.jl")
include("plotData.jl")
include("RNA.jl")
include("rnaUtils.jl")
include("utils.jl")
include("execute.jl")

install = true;