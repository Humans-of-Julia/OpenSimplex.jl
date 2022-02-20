const ROOT3OVER3 = 0.577350269189626
const FALLBACK_ROTATE3 = 2.0 / 3.0
const ROTATE3_ORTHOGONALIZER = UNSKEW_2D

const RSQUARED_3D = 3.0f0 / 4.0f0

export ImproveXY, ImproveXZ

struct ImproveXY <: NoiseType end
struct ImproveXZ <: NoiseType end

"""
3D OpenSimplex2S/SuperSimplex noise, with better visual isotropy in (X, Y).
Recommended for 3D terrain and time-varied animations.
The Z coordinate should always be the "different" coordinate in whatever your use case is.
If Y is vertical in world coordinates, call `noise(ImproveXZ, seed, x, z, Y)`.
If Z is vertical in world coordinates, call `noise(ImproveXZ, seed, x, y, Z)`.
For a time varied animation, call `noise(ImproveXY, seed, x, y, T)`.
"""
function noise(::Type{ImproveXY}, seed::Int64, x, y, z)
    # Re-orient the cubic lattices without skewing, so Z points up the main lattice diagonal,
    # and the planes formed by XY are moved far out of alignment with the cube faces.
    # Orthonormal rotation. Not a skew transform.
    xy = x + y
    s2 = xy * ROTATE3_ORTHOGONALIZER
    zz = z * ROOT3OVER3

    # Evaluate both lattices to form a BCC lattice.
    return _noise(seed, x + s2 + zz, y + s2 + zz, xy * -ROOT3OVER3 + zz)
end

"""
3D OpenSimplex2S/SuperSimplex noise, with better visual isotropy in (X, Z).
Recommended for 3D terrain and time-varied animations.
The Y coordinate should always be the "different" coordinate in whatever your use case is.
If Y is vertical in world coordinates, call `noise(ImproveXZ, seed, x, Y, z)`.
If Z is vertical in world coordinates, call `noise(ImproveXZ, seed, x, Z, y)`
or use `noise(ImproveXY, ...)`
For a time varied animation, call `noise(ImproveXZ, seed, x, T, y)`
or use `noise(ImproveXY, ...)`
"""
function noise(::Type{ImproveXZ}, seed::Int64, x, y, z)
    # Re-orient the cubic lattices without skewing, so Y points up the main lattice diagonal,
    # and the planes formed by XZ are moved far out of alignment with the cube faces.
    # Orthonormal rotation. Not a skew transform.
    xz = x + z
    s2 = xz * -0.211324865405187
    yy = y * ROOT3OVER3
    # Evaluate both lattices to form a BCC lattice.
    return _noise(seed, x + s2 + yy, xz * -ROOT3OVER3 + yy, z + s2 + yy)
end

"""
3D OpenSimplex2S/SuperSimplex noise, fallback rotation option
Use `noise(ImproveXY,...)` or `noise(ImproveXZ,...)` instead, wherever appropriate.
They have less diagonal bias. This function's best use is as a fallback.
"""
function noise(seed::Int64, x, y, z)
    # Re-orient the cubic lattices via rotation, to produce a familiar look.
    # Orthonormal rotation. Not a skew transform.
    r = FALLBACK_ROTATE3 * (x + y + z)
    # Evaluate both lattices to form a BCC lattice.
    return _noise(seed, r - x, r - y, r - z)
end

toint(x) = trunc(Int, -0.5f0 - x)

"""
Generate overlapping cubic lattices for 3D Re-oriented BCC noise.
Lookup table implementation inspired by DigitalShadow.
It was actually faster to narrow down the points in the loop itself,
than to build up the index with enough info to isolate 8 points.
"""
function _noise(seed::Int64, xr, yr, zr)
    # Get base points and offsets.
    xrb, yrb, zrb = fastFloor(xr), fastFloor(yr), fastFloor(zr)
    xi, yi, zi = Float32(xr - xrb), Float32(yr - yrb), Float32(zr - zrb)

    # Prime pre-multiplication for hash. Also flip seed for second lattice copy.
    xrbp, yrbp, zrbp = (xrb * PRIME_X)%Int64, (yrb * PRIME_Y)%Int64, (zrb * PRIME_Z)%Int64
    seed2 = xor(seed, -0x52D547B2E96ED629)

    # -1 if positive, 0 if negative.
    xNMask, yNMask, zNMask = toint(xi), toint(yi), toint(zi)

    # First vertex.
    x0, y0, z0 = xi + xNMask, yi + yNMask, zi + zNMask
    a0 = RSQUARED_3D - x0 * x0 - y0 * y0 - z0 * z0
    value = grad(a0, seed,
                 xrbp + (xNMask & PRIME_X), yrbp + (yNMask & PRIME_Y), zrbp + (zNMask & PRIME_Z),
                 x0, y0, z0)

    # Second vertex.
    x1, y1, z1 = xi - 0.5f0, yi - 0.5f0, zi - 0.5f0
    a1 = RSQUARED_3D - x1 * x1 - y1 * y1 - z1 * z1
    value += grad(a1, seed2, xrbp + PRIME_X, yrbp + PRIME_Y, zrbp + PRIME_Z, x1, y1, z1)

    # Shortcuts for building the remaining falloffs.
    # Derived by subtracting the polynomials with the offsets plugged in.
    xAFlipMask0 = ((xNMask | 1) << 1) * x1
    yAFlipMask0 = ((yNMask | 1) << 1) * y1
    zAFlipMask0 = ((zNMask | 1) << 1) * z1
    xAFlipMask1 = (-2 - (xNMask << 2)) * x1 - 1.0f0
    yAFlipMask1 = (-2 - (yNMask << 2)) * y1 - 1.0f0
    zAFlipMask1 = (-2 - (zNMask << 2)) * z1 - 1.0f0

    skip5 = false
    a = xAFlipMask0 + a0
    if a > 0
        value += grad(a, seed,
                      xrbp + (~xNMask & PRIME_X),
                      yrbp + (yNMask & PRIME_Y),
                      zrbp + (zNMask & PRIME_Z),
                      x0 - (xNMask | 1), y0, z0)
    else
        a = yAFlipMask0 + zAFlipMask0 + a0
        if a > 0
            value += grad(a, seed,
                          xrbp + (xNMask & PRIME_X),
                          yrbp + (~yNMask & PRIME_Y),
                          zrbp + (~zNMask & PRIME_Z),
                          x0, y0 - (yNMask | 1), z0 - (zNMask | 1))
        end

        a = xAFlipMask1 + a1
        if a > 0
            value += grad(a, seed2,
                          xrbp + (xNMask & (PRIME_X * 2)),
                          yrbp + PRIME_Y,
                          zrbp + PRIME_Z,
                          (xNMask | 1) + x1, y1, z1)
            skip5 = true
        end
    end

    skip9 = false
    a = yAFlipMask0 + a0
    if a > 0
        value += grad(a, seed,
                      xrbp + (xNMask & PRIME_X),
                      yrbp + (~yNMask & PRIME_Y),
                      zrbp + (zNMask & PRIME_Z),
                      x0, y0 - (yNMask | 1), z0)
    else
        a = xAFlipMask0 + zAFlipMask0 + a0
        if a > 0
            value += grad(a, seed,
                          xrbp + (~xNMask & PRIME_X),
                          yrbp + (yNMask & PRIME_Y),
                          zrbp + (~zNMask & PRIME_Z),
                          x0 - (xNMask | 1), y0, z0 - (zNMask | 1))
        end

        a = yAFlipMask1 + a1
        if a > 0
            value += grad(a, seed2,
                          xrbp + PRIME_X,
                          yrbp + (yNMask & (PRIME_Y << 1)),
                          zrbp + PRIME_Z,
                          x1, (yNMask | 1) + y1, z1)
            skip9 = true
        end
    end

    skipD = false
    a = zAFlipMask0 + a0
    if a > 0
        value += grad(a, seed,
                      xrbp + (xNMask & PRIME_X),
                      yrbp + (yNMask & PRIME_Y),
                      zrbp + (~zNMask & PRIME_Z),
                      x0, y0, z0 - (zNMask | 1))
    else
        a = xAFlipMask0 + yAFlipMask0 + a0
        if a > 0
            value += grad(a, seed,
                          xrbp + (~xNMask & PRIME_X),
                          yrbp + (~yNMask & PRIME_Y),
                          zrbp + (zNMask & PRIME_Z),
                          x0 - (xNMask | 1), y0 - (yNMask | 1), z0)
        end

        a = zAFlipMask1 + a1
        if a > 0
            value += grad(a, seed2,
                          xrbp + PRIME_X,
                          yrbp + PRIME_Y,
                          zrbp + (zNMask & (PRIME_Z << 1)),
                          x1, y1, (zNMask | 1) + z1)
            skipD = true
        end
    end

    if !skip5
        a = yAFlipMask1 + zAFlipMask1 + a1
        if a > 0
            value += grad(a, seed2,
                          xrbp + PRIME_X,
                          yrbp + (yNMask & (PRIME_Y << 1)),
                          zrbp + (zNMask & (PRIME_Z << 1)),
                          x1, (yNMask | 1) + y1, (zNMask | 1) + z1)
        end
    end

    if !skip9
        a = xAFlipMask1 + zAFlipMask1 + a1
        if a > 0
            value += grad(a, seed2,
                          xrbp + (xNMask & (PRIME_X * 2)),
                          yrbp + PRIME_Y,
                          zrbp + (zNMask & (PRIME_Z << 1)),
                          (xNMask | 1) + x1, y1, (zNMask | 1) + z1)
        end
    end

    if !skipD
        a = xAFlipMask1 + yAFlipMask1 + a1
        if a > 0
            value += grad(a, seed2,
                          xrbp + (xNMask & (PRIME_X << 1)),
                          yrbp + (yNMask & (PRIME_Y << 1)),
                          zrbp + PRIME_Z,
                          (xNMask | 1) + x1, (yNMask | 1) + y1, z1)
        end
    end

    return Float32(value)
end
