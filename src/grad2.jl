const N_GRADS_2D_EXPONENT = 7
const N_GRADS_2D = 1 << N_GRADS_2D_EXPONENT

const NORMALIZER_2D = 0.05481866495625118

struct Grad2
    x::Float32
    y::Float32
end

const grad2 = Grad2[
    Grad2( 0.38268343236509,   0.923879532511287),
    Grad2( 0.923879532511287,  0.38268343236509),
    Grad2( 0.923879532511287, -0.38268343236509),
    Grad2( 0.38268343236509,  -0.923879532511287),
    Grad2(-0.38268343236509,  -0.923879532511287),
    Grad2(-0.923879532511287, -0.38268343236509),
    Grad2(-0.923879532511287,  0.38268343236509),
    Grad2(-0.38268343236509,   0.923879532511287),
    #-------------------------------------------#
    Grad2( 0.130526192220052,  0.99144486137381),
    Grad2( 0.608761429008721,  0.793353340291235),
    Grad2( 0.793353340291235,  0.608761429008721),
    Grad2( 0.99144486137381,   0.130526192220051),
    Grad2( 0.99144486137381,  -0.130526192220051),
    Grad2( 0.793353340291235, -0.60876142900872),
    Grad2( 0.608761429008721, -0.793353340291235),
    Grad2( 0.130526192220052, -0.99144486137381),
    Grad2(-0.130526192220052, -0.99144486137381),
    Grad2(-0.608761429008721, -0.793353340291235),
    Grad2(-0.793353340291235, -0.608761429008721),
    Grad2(-0.99144486137381,  -0.130526192220052),
    Grad2(-0.99144486137381,   0.130526192220051),
    Grad2(-0.793353340291235,  0.608761429008721),
    Grad2(-0.608761429008721,  0.793353340291235),
    Grad2(-0.130526192220052,  0.99144486137381),
]

const GRADIENTS_2D = Vector{Grad2}(undef, N_GRADS_2D)

function grad(a, seed, xsvp, ysvp, dx, dy)
    hash = xor(seed, xsvp, ysvp) * HASH_MULTIPLIER
    g = GRADIENTS_2D[(xor(hash, hash >> (64 - N_GRADS_2D_EXPONENT)) & (N_GRADS_2D - 1)) + 1]
    Float32(p4(a) * (g.x * dx + g.y * dy) / NORMALIZER_2D)
end

function fill_2d()
    j = 0
    len = length(grad2)
    for i = 1:N_GRADS_2D
        GRADIENTS_2D[i] = grad2[j += 1]
        j == len && (j = 0)
    end
end

fill_2d()
