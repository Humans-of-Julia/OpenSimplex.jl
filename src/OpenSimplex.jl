"""
Port of K.jpg's OpenSimplex 2, smooth variant ("SuperSimplex") to Julia
"""
module OpenSimplex

const PRIME_X = 0x5205402B9270C86F
const PRIME_Y = 0x598CD327003817B5
const PRIME_Z = 0x5BCC226E9FA0BACB
const PRIME_W = 0x56CC5227E58F554B
const HASH_MULTIPLIER = 0x53A3F72DEEC546F5

abstract type NoiseType end

p4(v) = (v2 = v * v ; v2 * v2)

# constants & lookup tables

include("grad2.jl")
include("grad3.jl")
include("grad4.jl")

# Noise Evaluators

export noise

function noise end

include("noise2d.jl")
include("noise3d.jl")
include("noise4d.jl")
include("vertexcodes.jl")

# Utility

#=
function fastFloor(x::Float64)
    xi = Int64(x)
    return x < xi ? xi - 1 : xi
end
=#
fastFloor(x::Float64) = floor(Int64, x)
end # module OpenSimplex
