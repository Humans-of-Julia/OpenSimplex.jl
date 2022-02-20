const SKEW_4D = 0.309016994374947f0
const UNSKEW_4D = -0.138196601125011f0

const RSQUARED_4D = 4.0f0 / 5.0f0

const S2VAL = -0.211324865405187 # (1/sqrt(N+1)-1)/N where N=2
const WWMUL = 1.118033988749894
const WWVAL = -0.866025403784439
const YZVAL = 0.28867513459481294226
const INVSQRT3 = -0.5773502691896257 # (-1/sqrt(3))

export ImproveXYZ, ImproveXYZ_XY, ImproveXYZ_XZ, ImproveXY_ZW
struct ImproveXYZ_XY <: NoiseType end
struct ImproveXYZ_XZ <: NoiseType end
struct ImproveXYZ    <: NoiseType end
struct ImproveXY_ZW  <: NoiseType end

"""
4D SuperSimplex noise, with XYZ oriented like `noise(ImproveXY, ...)`
and W for an extra degree of freedom. W repeats eventually.
Recommended for time-varied animations which texture a 3D object (W=time)
in a space where Z is vertical
"""
function noise(::Type{ImproveXYZ_XY}, seed::Int64, x, y, z, w)
    xy = x + y
    s2 = xy * S2VAL
    zz = z * YZVAL
    ww = w * WWMUL
    r = zz + ww + s2
    _noise(seed, x + r, y + r, xy * INVSQRT3 + (zz + ww), z * WWVAL + ww)
end

"""
4D SuperSimplex noise, with XYZ oriented like `noise(ImproveXZ, ...)`
and W for an extra degree of freedom. W repeats eventually.
Recommended for time-varied animations which texture a 3D object (W=time)
in a space where Y is vertical
"""
function noise(::Type{ImproveXYZ_XZ}, seed::Int64, x, y, z, w)
    xz = x + z
    s2 = xz * S2VAL
    yy = y * YZVAL
    ww = w * WWMUL
    r = yy + ww + s2
    _noise(seed, x + r, xz * INVSQRT3 + (yy + ww), z + r, y * WWVAL + ww)
end

"""
4D SuperSimplex noise, with XYZ oriented like 3D `noise`
and W for an extra degree of freedom. W repeats eventually.
Recommended for time-varied animations which texture a 3D object (W=time)
where there isn't a clear distinction between horizontal and vertical
"""
function noise(::Type{ImproveXYZ}, seed::Int64, x, y, z, w)
    xyz = x + y + z
    ww = w * WWMUL
    s2 = xyz * -0.16666666666666666 + ww
    _noise(seed, x + s2, y + s2, z + s2, -0.5 * xyz + ww)
end
    
"""
4D SuperSimplex noise, with XY and ZW forming orthogonal triangular-based planes.
Recommended for 3D terrain, where X and Y (or Z and W) are horizontal.
Recommended for `noise(seed, x, y, sin(time), cos(time))` trick.
"""
function noise(::Type{ImproveXY_ZW}, seed::Int64, x, y, z, w)
    s2 = (x + y) * -0.28522513987434876941 + (z + w) * 0.83897065470611435718
    t2 = (z + w) * 0.21939749883706435719 + (x + y) * -0.48214856493302476942
    return _noise(seed, x + s2, y + s2, z + t2, w + t2)
end

"""
4D SuperSimplex noise, fallback lattice orientation.
"""
function noise(seed::Int64, x, y, z, w)
    # Get points for A4 lattice
    s = SKEW_4D * (x + y + z + w)
    return _noise(seed, x + s, y + s, z + s, w + s)
end

"""
4D SuperSimplex noise base.
Using ultra-simple 4x4x4x4 lookup partitioning.
This isn't as elegant or SIMD/GPU/etc. portable as other approaches,
but it competes performance-wise with optimized 2014 OpenSimplex.
"""
function _noise(seed::Int64, xs, ys, zs, ws)
    # Get base points and offsets
    xsb, ysb, zsb, wsb = fastFloor(xs), fastFloor(ys), fastFloor(zs), fastFloor(ws)
    xsi, ysi, zsi, wsi = Float32(xs - xsb), Float32(ys - ysb), Float32(zs - zsb), Float32(ws - wsb)

    # Unskewed offsets
    ssi = (xsi + ysi + zsi + wsi) * UNSKEW_4D
    xi, yi, zi, wi = xsi + ssi, ysi + ssi, zsi + ssi, wsi + ssi

    # Prime pre-multiplication for hash.
    xsvp = (xsb * PRIME_X)%Int64
    ysvp = (ysb * PRIME_Y)%Int64
    zsvp = (zsb * PRIME_Z)%Int64
    wsvp = (wsb * PRIME_W)%Int64

    # Index into initial table.
    index = ((fastFloor(xs * 4) & 3) << 0) |
            ((fastFloor(ys * 4) & 3) << 2) |
            ((fastFloor(zs * 4) & 3) << 4) |
            ((fastFloor(ws * 4) & 3) << 6)

    # Point contributions
    value = 0f0
    ind_beg = index == 0 ? 1 : Int(offsets_4D[index])
    ind_end = Int(offsets_4D[index+1])
    for i = ind_beg:ind_end
        c = lookup_4D[vertex_ind[i]]
        dx, dy, dz, dw = xi + c.dx, yi + c.dy, zi + c.dz, wi + c.dw
        a = (dx * dx + dy * dy) + (dz * dz + dw * dw)
        if a < RSQUARED_4D
            value += grad(a - RSQUARED_4D, seed,
                          xsvp + c.xsvp,
                          ysvp + c.ysvp,
                          zsvp + c.zsvp,
                          wsvp + c.wsvp,
                          dx, dy, dz, dw)
        end
    end
    return Float32(value)
end
