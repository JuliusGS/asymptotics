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

"""
    final_advection_distance(p::Params; X0=0.0, z0=0.0, atol=1e-12, rtol=1e-8) -> Float64

Total horizontal advection distance the slow-manifold model predicts for a particle
released at `(x, z) = (X0, z0)` (with `t = 0`, so the wave-frame coordinate is `X0 = x0`):

    d_fin = ∫ (1 + 1/f(X')) dX'   from X0 toward X → −∞,

where `f(X) = xdot(p)(X, z(X)) = -1 + eᶻ Re(A e^{iX})` is the wave-frame slow-manifold
velocity along the isobar `z(X)` through the release point (`c = H(X0, z0)`,
`A = amplitude(p)`). Since `f < 0` (the particle settles, `Ẋ < 0`), `X` decreases from
`X0`; the integrand decays like `e^{vz·X}` as `z → −∞`, so the distance is finite.

`z(X)` is the isobar `H(X,z) = c`, obtained by root finding — the same curve as the
closed form `z(X) = vz·X − c − W₀(−Re(A e^{iX}) e^{vz·X − c})`. Released from rest at the
origin (`X0 = z0 = 0`) gives `c = Re(A)`.

For `b = 1` the amplitude collapses to `A = ε` (the inertialess `G`), so the distance
matches the inertialess prediction; for `b ≠ 1` it differs — that difference is the
inertial correction the model captures.
"""
function final_advection_distance(p::Params; X0=0.0, z0=0.0, atol=1e-12, rtol=1e-8)
    vz = p.vz
    vz > 0 || throw(ArgumentError("final_advection_distance needs vz > 0 (a settling particle)"))
    A = amplitude(p)
    H = hamiltonian(p); Xd = xdot(p)
    c = H(X0, z0)

    # isobar z(X): H(X,·)=c. Monotone in z on the settling branch, and the closed form
    # z = vz*X - c - W₀(…) with W₀ ≥ -1 pins z ∈ [vz*X-c-2, vz*X-c+1] — a guaranteed
    # bracket, so use a bracketing solve (robust for any release phase X0).
    zof(X)       = (zc = vz*X - c; find_zero(z -> H(X, z) - c, (zc - 2.0, zc + 1.001)))
    integrand(X) = 1 + 1/Xd(X, zof(X))                     # 1 + 1/f(X)

    # f → −1 and the integrand's envelope ~ |A| e^{vz·X − c} decays as X → −∞;
    # integrate to where that envelope drops below atol.
    Xfar = min((c + log(atol) - log(abs(A))) / vz, X0 - 1.0)
    return quadgk(integrand, X0, Xfar; rtol=rtol, atol=atol)[1]
end

"""
    final_advection_distance_mr(p::Params; X0=0.0, z0=0.0, T=30/p.vz,
                                reltol=1e-10, abstol=1e-12) -> Float64

Same quantity as [`final_advection_distance`](@ref) but for the **full Maxey–Riley**
system ([`maxey_riley!`](@ref)) instead of the slow-manifold reconstruction: integrate
the particle ODE until it has settled (`z ≈ -30`) and return the net horizontal
displacement `x(T) − X0`. Released ON the slow manifold at `(X0, z0)`
(`v = A·E₀ − i·vz`, `A = amplitude(p)`) so there is no from-rest transient.

Uses the decoupled convention (`vz`, `St` independent), like everything else here.
Compare `final_advection_distance_mr(p) − final_advection_distance_mr(p_{b=1})` against
the slow-manifold correction to see how much of the O(St) effect the reduction captures.
"""
function final_advection_distance_mr(p::Params; X0=0.0, z0=0.0, T=nothing,
                                     reltol=1e-10, abstol=1e-12)
    vz = p.vz
    vz > 0 || throw(ArgumentError("final_advection_distance_mr needs vz > 0"))
    A  = amplitude(p)
    E0 = exp(z0 + im*X0)
    u0 = [X0, z0, real(A*E0), imag(A*E0) - vz]        # on-manifold release
    Tset = T === nothing ? 30/vz : T                  # settle to z ≈ -30 (drift saturated)
    sol = solve(ODEProblem(maxey_riley!, u0, (0.0, Tset), p),
                abstol=abstol, reltol=reltol, save_everystep=false, maxiters=1e9)
    return sol.u[end][1] - X0
end

"""
    final_advection_distance_fs(p::Params; X0=0.0, z0=0.0, T=30/p.vz,
                                reltol=1e-10, abstol=1e-12) -> Float64

Final advection distance of the **fluid + settling** model ([`fluid_settling!`](@ref)) —
a fluid element that also settles at `vz`. This is the exact **St = 0 (no-inertia)
limit**, depending only on `eps` and `vz` (not `St` or `b`). Integrate `fluid_settling!`
until settled and return the net horizontal displacement `x(T) − X0`.

It coincides (to solver tolerance) with `final_advection_distance(p_{b=1})`: the reduced
model's `b = 1` distance equals this because `fluid_settling!` exactly conserves the
`A = ε` Hamiltonian; and the full-M–R `b = 1` particle equals it too. It is therefore the
natural common baseline for the inertial correction of *either* description.
"""
function final_advection_distance_fs(p::Params; X0=0.0, z0=0.0, T=nothing,
                                     reltol=1e-10, abstol=1e-12)
    vz = p.vz
    vz > 0 || throw(ArgumentError("final_advection_distance_fs needs vz > 0"))
    Tset = T === nothing ? 30/vz : T
    sol = solve(ODEProblem(fluid_settling!, [X0, z0], (0.0, Tset), p),
                abstol=abstol, reltol=reltol, save_everystep=false, maxiters=1e9)
    return sol.u[end][1] - X0
end

"""
    period_averaged_start(p::Params; u0=[0,0,ε,0], t_scan=8π, dt=0.01,
                          reltol=1e-10, abstol=1e-12) -> (; avgd_x0, avgd_z0, ntptimes)

Phase-averaged "true" starting point of a full Maxey–Riley trajectory. Solve from `u0`,
find the first two times the horizontal velocity `vₓ = w[3]` crosses zero from negative to
positive (`ntptimes`, one oscillation period), and average the position over that period.
This removes the arbitrary instantaneous release phase/transient. Follows the
`just_plot_z_vs_t` construction in the 2026.03 `plot_asymptotics.jl`.
"""
# Average (x, z) of a dense solution over the first oscillation period, delimited by the
# first two negative→positive crossings of the horizontal velocity `vx(t)`.
function _avg_first_period(sol, vx; t_scan, dt)
    ntp = Float64[]
    tg  = dt:dt:t_scan
    for i in 1:length(tg)-1
        (vx(tg[i]) < 0 <= vx(tg[i+1])) || continue
        push!(ntp, find_zero(vx, (tg[i], tg[i+1]), Bisection()))
        length(ntp) == 2 && break
    end
    length(ntp) == 2 ||
        error("_avg_first_period: found $(length(ntp)) neg→pos vₓ crossings in [0, $t_scan]")
    ts = ntp[1]:dt:ntp[2]
    return (; avgd_x0 = mean(sol(t)[1] for t in ts),
              avgd_z0 = mean(sol(t)[2] for t in ts),
              ntptimes = (ntp[1], ntp[2]))
end

function period_averaged_start(p::Params; dynamics=maxey_riley!, u0=nothing, t_scan=8pi,
                               dt=0.01, reltol=1e-10, abstol=1e-12)
    ic  = u0 === nothing ? [0.0, 0.0, p.eps, 0.0] : u0
    sol = solve(ODEProblem(dynamics, ic, (0.0, t_scan), p),
                abstol=abstol, reltol=reltol, dense=true, maxiters=1e9)
    return _avg_first_period(sol, t -> sol(t)[3]; t_scan=t_scan, dt=dt)  # vₓ = w[3]
end

"""
    final_advection_distance_mr_avgstart(p::Params; u0=[0,0,ε,0], T=30/vz,
                                         reltol=1e-10, abstol=1e-12) -> Float64

Final advection distance of the full Maxey–Riley particle measured from the phase-averaged
"true" start ([`period_averaged_start`](@ref)) instead of the instantaneous release point:
`x(T) − avgd_x0`. Averaging out the first-period oscillation removes the O(St) boundary
offset, isolating the drift displacement.
"""
function final_advection_distance_mr_avgstart(p::Params; dynamics=maxey_riley!, u0=nothing,
                                              T=nothing, reltol=1e-10, abstol=1e-12)
    ic = u0 === nothing ? [0.0, 0.0, p.eps, 0.0] : u0
    s  = period_averaged_start(p; dynamics=dynamics, u0=ic, reltol=reltol, abstol=abstol)
    Tset = T === nothing ? 30/p.vz : T
    sol = solve(ODEProblem(dynamics, ic, (0.0, Tset), p),
                abstol=abstol, reltol=reltol, save_everystep=false, maxiters=1e9)
    return sol.u[end][1] - s.avgd_x0
end

"""
    final_advection_distance_fs_avgstart(p::Params; t_scan=8π, dt=0.01, T=30/vz,
                                         reltol=1e-10, abstol=1e-12) -> Float64

St = 0 limit of [`final_advection_distance_mr_avgstart`](@ref): the fluid+settling
([`fluid_settling!`](@ref)) final advection distance measured from its own phase-averaged
start (first period of horizontal velocity `vₓ = ε eᶻ cos(x − t)`). Depends only on `eps`
and `vz`. This is the baseline the M–R averaged-start distance approaches as St → 0.
"""
function final_advection_distance_fs_avgstart(p::Params; t_scan=8pi, dt=0.01, T=nothing,
                                              reltol=1e-10, abstol=1e-12)
    eps = p.eps
    sol = solve(ODEProblem(fluid_settling!, [0.0, 0.0], (0.0, t_scan), p),
                abstol=abstol, reltol=reltol, dense=true, maxiters=1e9)
    vx(t) = eps*exp(sol(t)[2])*cos(sol(t)[1] - t)
    s = _avg_first_period(sol, vx; t_scan=t_scan, dt=dt)
    Tset = T === nothing ? 30/p.vz : T
    solE = solve(ODEProblem(fluid_settling!, [0.0, 0.0], (0.0, Tset), p),
                 abstol=abstol, reltol=reltol, save_everystep=false, maxiters=1e9)
    return solE.u[end][1] - s.avgd_x0
end
