const ROOT2OVER2 = 0.7071067811865476
const SKEW_2D = 0.366025403784439
const UNSKEW_2D = -0.21132486540518713

const RSQUARED_2D = 2.0f0 / 3.0f0

export ImproveX
struct ImproveX <: NoiseType end

"""
2D OpenSimplex2S/SuperSimplex noise, standard lattice orientation
"""
function noise(seed::Int64, x, y)
    # Get points for A2* lattice
    s = SKEW_2D * (x + y)
    _noise(seed, x + s, y + s)
end

"""
2D OpenSimplex2S/SuperSimplex noise, with Y pointing down the main diagonal.
Might be better for a 2D sandbox style game, where Y is vertical.
Probably slightly less optimal for heightmaps or continent maps,
unless your map is centered around an equator. It's a slight
difference, but the option is here to make it easy.
"""
function noise(::Type{ImproveX}, seed::Int64, x, y)
    # Skew transform and rotation baked into one.
    xx = x * ROOT2OVER2
    yy = y * (ROOT2OVER2 * (1 + 2 * SKEW_2D))
    _noise(seed, yy + xx, yy - xx)
end

dxy(x, y) = RSQUARED_2D - x * x - y * y

"""
2D  OpenSimplex2S/SuperSimplex noise base.
"""
function _noise(seed::Int64, xs, ys)

    # Get base points and offsets.
    xsb = fastFloor(xs)
    ysb = fastFloor(ys)
    xi = Float32(xs - xsb)
    yi = Float32(ys - ysb)

    # Prime pre-multiplication for hash.
    xsbp = (xsb * PRIME_X)%Int64
    ysbp = (ysb * PRIME_Y)%Int64

    # Unskew.
    t = (xi + yi) * Float32(UNSKEW_2D)
    dx0 = xi + t
    dy0 = yi + t

    UNSKEW_2D_2X = (1 + 2 * UNSKEW_2D)

    # First vertex.
    a0 = dxy(dx0, dy0)
    value = grad(a0, seed, xsbp, ysbp, dx0, dy0)

    # Second vertex.
    a = Float32(2 * UNSKEW_2D_2X * (1 / UNSKEW_2D + 2)) * t +
        (Float32(-2 * UNSKEW_2D_2X * UNSKEW_2D_2X) + a0)
    value += grad(a, seed, xsbp + PRIME_X, ysbp + PRIME_Y,
                  dx0 - Float32(UNSKEW_2D_2X), dy0 - Float32(UNSKEW_2D_2X))

    # Third and fourth vertices.
    # Nested conditionals were faster than compact bit logic/arithmetic.
    xmyi = xi - yi
    if t < UNSKEW_2D
        if xi + xmyi > 1
            dx = dx0 - Float32(3 * UNSKEW_2D + 2)
            dy = dy0 - Float32(3 * UNSKEW_2D + 1)
            a = dxy(dx, dy)
            if a > 0
                value += grad(a, seed, xsbp + (PRIME_X << 1), ysbp + PRIME_Y, dx, dy)
            end
        else
            dx = dx0 - Float32(UNSKEW_2D)
            dy = dy0 - Float32(UNSKEW_2D + 1)
            a = dxy(dx, dy)
            if a > 0
                value += grad(a, seed, xsbp, ysbp + PRIME_Y, dx, dy)
            end
        end

        if yi - xmyi > 1
            dx = dx0 - Float32(3 * UNSKEW_2D + 1)
            dy = dy0 - Float32(3 * UNSKEW_2D + 2)
            a = dxy(dx, dy)
            if a > 0
                value += grad(a, seed, xsbp + PRIME_X, ysbp + (PRIME_Y << 1), dx, dy)
            end
        else
            dx = dx0 - Float32(UNSKEW_2D + 1)
            dy = dy0 - Float32(UNSKEW_2D)
            a = dxy(dx, dy)
            if a > 0
                value += grad(a, seed, xsbp + PRIME_X, ysbp, dx, dy)
            end
        end
    else
        if xi + xmyi < 0
            dx = dx0 + Float32(1 + UNSKEW_2D)
            dy = dy0 + Float32(UNSKEW_2D)
            a = dxy(dx, dy)
            if a > 0
                value += grad(a, seed, xsbp - PRIME_X, ysbp, dx, dy)
            end
        else
            dx = dx0 - Float32(UNSKEW_2D + 1)
            dy = dy0 - Float32(UNSKEW_2D)
            a = dxy(dx, dy)
            if a > 0
                value += grad(a, seed, xsbp + PRIME_X, ysbp, dx, dy)
            end
        end

        if yi < xmyi
            dx = dx0 + Float32(UNSKEW_2D)
            dy = dy0 + Float32(UNSKEW_2D + 1)
            a = dxy(dx, dy)
            if a > 0
                value += grad(a, seed, xsbp, ysbp - PRIME_Y, dx, dy)
            end
        else
            dx = dx0 - Float32(UNSKEW_2D)
            dy = dy0 - Float32(UNSKEW_2D + 1)
            a = dxy(dx, dy)
            if a > 0
                value += grad(a, seed, xsbp, ysbp + PRIME_Y, dx, dy)
            end
        end
    end

    return Float32(value)
end
