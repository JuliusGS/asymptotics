# Lab-frame trajectory reconstruction from a single isobar — no particle ODE solved.

"""
    recover_xzt(p::Params; X0=0.0, z0=0.0, Xspan=5/p.vz, N=4000) -> (; x, z, t, X)

Reconstruct a lab-frame trajectory `(x, z)` from the single isobar
`H(X, z) = H(X0, z0)` through the release point, WITHOUT integrating the particle ODE:

1. `z(X)` from the isobar (`find_zero`; `H` is monotone in `z`), warm-started along `X`;
2. `t(X) = ∫ dX / Xdot(X, z(X))`   (`quadgk`, accumulated segment by segment);
3. lab-frame position `x = X + t(X)`.

`X` runs from `X0` down to `X0 - Xspan`. Returns named arrays `x, z, t, X`.
"""
function recover_xzt(p::Params; X0=0.0, z0=0.0, Xspan=5/p.vz, N=4000)
    H = hamiltonian(p); Xd = xdot(p)
    c = H(X0, z0)
    zprev = Ref(z0)
    zof(X) = (zprev[] = find_zero(zz -> H(X, zz) - c, zprev[]); zprev[])

    X = range(X0, X0 - Xspan, length=N)
    z = zof.(X)
    t = zeros(N)
    for k in 2:N
        t[k] = t[k-1] + quadgk(Xp -> 1/Xd(Xp, zof(Xp)), X[k-1], X[k])[1]
    end
    x = X .+ t
    return (; x = collect(x), z = collect(z), t = t, X = collect(X))
end
