# Deterministic right-hand sides ("det-functions") for `ODEProblem`.
#
# All are in the v_z convention (see `Params`): the settling velocity is `p.vz`, and
# the Maxey–Riley gravity term is `-vz/St`, so the particle settles at exactly `vz`
# in still fluid, independent of St.
#
# These are vendored copies of the functions actually used from the 2026.03 codebase,
# with the convention standardised to v_z:
#   * maxey_riley!    == santamaria_full_eq_added_mass_new_dudt_material_settling_am_separate!
#   * fluid_settling! == particle_is_fluid_and_settling_inf_sep_settling_ad!  (the `-vz` form)
# so the fragile external include chain (classes.jl -> ODEs_new.jl -> plot_asymptotics.jl)
# is dropped entirely.

"""
    maxey_riley!(dw, w, p::Params, t)

Full decoupled Maxey–Riley system with settling and added mass separated.
State `w = [x, z, vx, vz]`; the fluid velocity is the deep-water wave
`u = eᶻ (cos(x-t), sin(x-t))`. Gravity enters as `-vz/St` (v_z convention), so the
terminal settling velocity in still fluid is exactly `p.vz`.
"""
function maxey_riley!(dw, w, p::Params, t)
    St = p.St; eps = p.eps; b = p.b; vz = p.vz
    u1 = exp(w[2]) * cos(w[1] - t)
    u2 = exp(w[2]) * sin(w[1] - t)
    dw[1] = w[3]
    dw[2] = w[4]
    dw[3] = (eps*u1 - w[3])/St + b*eps*(u2 - w[3]*u2 + w[4]*u1)
    dw[4] = (eps*u2 - w[4])/St - vz/St + b*eps*(-u1 + w[3]*u1 + w[4]*u2)
end

"""
    maxey_riley_DDt!(dw, w, p::Params, t)

DiBenedetto's form of the decoupled Maxey–Riley system: identical to [`maxey_riley!`](@ref)
except the added-mass term uses the **fluid** material derivative `Du/Dt` (following the
fluid, Auton–Hunt–Prud'Homme 1988) rather than the particle-following `du/dt`. For the
deep-water wave `Du₁/Dt = u₂` and `Du₂/Dt = -u₁ + ε e^{2z}` (the advection cancels in x
and contributes an `ε e^{2z}` mean term in z). Use to test whether a drift feature is
sensitive to the d/dt-vs-D/Dt modelling choice.
"""
function maxey_riley_DDt!(dw, w, p::Params, t)
    St = p.St; eps = p.eps; b = p.b; vz = p.vz
    u1 = exp(w[2]) * cos(w[1] - t)
    u2 = exp(w[2]) * sin(w[1] - t)
    dw[1] = w[3]
    dw[2] = w[4]
    dw[3] = (eps*u1 - w[3])/St + b*eps*u2                          # + b ε Du₁/Dt
    dw[4] = (eps*u2 - w[4])/St - vz/St + b*eps*(-u1 + eps*exp(2*w[2]))  # + b ε Du₂/Dt
end

"""
    fluid_settling!(dw, w, p::Params, t)

Inertialess reference: the particle is a fluid element that also settles at `vz`.
State `w = [x, z]`. The Stokes drift emerges from advection (it is NOT prescribed).
This is the `St → 0` (tracer + settling) limit of [`maxey_riley!`](@ref).
"""
function fluid_settling!(dw, w, p::Params, t)
    eps = p.eps; vz = p.vz
    dw[1] = eps*exp(w[2]) * cos(w[1] - t)
    dw[2] = eps*exp(w[2]) * sin(w[1] - t) - vz
end

"""
    sigma_omega!(dw, w, p::Params, t)

The reduced `(σ, ω)` system, complex, in the state `w = [σ, σ̄, ω, ω̄]`. Obtained from
the Maxey–Riley equations by the change of variables `σ = eᶻ⁻ⁱ⁽ˣ⁻ᵗ⁾`,
`ω = (vx-1) - b ε eᶻcos(X) + i(vz - b ε eᶻsin(X))`.

v_z convention: the old `θ` (with terminal velocity `St*θ`) is `θ = vz/St` here, so the
constant forcing `-1/St ∓ iθ` becomes `-1/St ∓ i vz/St`.
"""
function sigma_omega!(dw, w, p::Params, t)
    St = p.St; eps = p.eps; b = p.b; vz = p.vz
    g = vz / St                       # = old θ (terminal velocity is St*θ = vz)
    dw[1] = -im*w[1]*w[3] - im*b*eps*w[1]*w[2]
    dw[2] =  im*w[2]*w[4] + im*b*eps*w[2]*w[1]
    dw[3] = (eps*(1-b)*w[2] - w[3])/St - 1/St - im*g
    dw[4] = (eps*(1-b)*w[1] - w[4])/St - 1/St + im*g
end

"""
    slow_manifold_xz!(dw, w, p::Params, t)

The slow manifold (`B₁ = B₂ = 0`) written directly as a 2-D `(x, z)` flow. On the
manifold `ω` is slaved to `B₃ = e^{z+iX}/conj(vhat)` (with `X = x - t`), giving
`d(x+iz)/dt = B₃ - i vz + b ε e^{z+iX}`. Provably identical to the
Hphys / z(X) / t(X) isobar reconstruction in [`recover_xzt`](@ref).
"""
function slow_manifold_xz!(dw, w, p::Params, t)
    vh = eigendata(p).vhat
    X  = w[1] - t
    B3 = exp(w[2] + im*X) / conj(vh)
    d  = B3 - im*p.vz + p.b*p.eps*exp(w[2] + im*X)   # dx/dt + i dz/dt
    dw[1] = real(d)
    dw[2] = imag(d)
end

"""
    wave_averaged_drift!(dw, w, p::Params, t)

Wave-averaged (Stokes-drift) equations for the settling particle in infinite depth
(the deep-water limit of the notebook asymptotics), state `w = [x, z]`, `b' = 1-b`:

    ẋ = ε² e^{2z} (1 - b'St² + 2 b'St·vz - vz²)
    ż = -vz - ε² e^{2z} (b'St + vz)

Integrating these self-consistently (z evolving) is the coupled asymptotic path. Holding
`z` at the leading path `z_* = z0 - vz t` instead makes `ẋ(t)` explicit and integrates to
the closed form `ε² e^{2 z0}/(2 vz) · (bracket)`.
"""
function wave_averaged_drift!(dw, w, p::Params, t)
    eps = p.eps; vz = p.vz; St = p.St; bp = 1 - p.b
    e2z = eps^2 * exp(2*w[2])
    dw[1] = e2z * (1 - bp*St^2 + 2*bp*St*vz - vz^2)
    dw[2] = -vz - e2z * (bp*St + vz)
end


"""
    slaved_amplitude(p::Params) -> ComplexF64
Constant `S = St(|A|² - b ε conj(A))` of the slaved (adiabatically-eliminated) model.
The extra lab-frame term beyond the `B₁,₂=0` flow is `-i S e^{2z}`; `real(S) > 0` for
`b < 1` is what breaks area preservation. `A = amplitude(p)`.
"""
slaved_amplitude(p::Params) = p.St*(abs2(amplitude(p)) - p.b*p.eps*conj(amplitude(p)))

"""
    xdot_slaved(p::Params) -> Function
Wave-frame horizontal velocity of the *slaved* flow,
`Ẋ = -1 + eᶻ Re(A e^{iX}) + Sᵢ e^{2z}`, i.e. [`xdot`](@ref) plus the drift correction.
Returns the closure `Ẋ(X, z)`.
"""
function xdot_slaved(p::Params)
    A = amplitude(p); Si = imag(slaved_amplitude(p))
    return (X, z) -> -1 + exp(z)*real(A*exp(im*X)) + Si*exp(2z)
end

"""
    hdot_drift(p::Params) -> Function
The EXACT drift law `Ḣ = ∇H · f_slaved = g e^{2z} - e^{3z} Re(S̄ A e^{iX})`, with the
mean-drift constant `g = Sᵣ + Sᵢ vz = St ε² (1-b) / ((1 - St vz)² + St²) > 0`.
This is the closure fed to the c-update — it is a *field*, nonzero on isobars; it is NOT
the (identically-zero) variation of H along an isobar. Returns `Ḣ(X, z)`.
"""
function hdot_drift(p::Params)
    A = amplitude(p); S = slaved_amplitude(p); vz = p.vz
    g = real(S) + imag(S)*vz
    return (X, z) -> g*exp(2z) - exp(3z)*real(conj(S)*A*exp(im*X))
end

"""
    invert_isobar(p::Params) -> Function
Solve `H(X, z) = c` for `z` at fixed `X` by Newton continuation (seeded from a nearby
`z`), tracing the physical branch. `∂_z H = -1 + eᶻ Re(A e^{iX}) < 0` throughout the
subsurface region, so the root is unique and Newton is unconditionally well-conditioned
— no Lambert-W branch bookkeeping. Returns `z(X, c, zguess)`.
"""
function invert_isobar(p::Params)
    A = amplitude(p); vz = p.vz
    return function (X, c, zguess)
        f = real(A*exp(im*X))                 # Re(A e^{iX}); independent of z
        z = zguess
        for _ in 1:60
            F  = -z + vz*X + exp(z)*f - c
            dF = -1 + exp(z)*f
            Δ  = F/dF
            z -= Δ
            abs(Δ) < 1e-13 && return z
        end
        return z
    end
end
"""
    slaved_isobar_xz(p::Params; X0, z0, nperiods, nquad, zmin)

Reconstruct the slaved-model trajectory by the per-period isobar scheme:

1. hold the label `c = H` FROZEN over one wave period `[Xₖ, Xₖ-2π]`;
2. build the frozen-`c` isobar `z(X, cₖ)` and integrate `dt/dX = 1/Ẋ` across the period;
3. update `c` ONCE, `c_{k+1} = cₖ + ∫ (Ḣ/Ẋ) dX`, using the analytic drift field
   [`hdot_drift`](@ref) — nonzero on the isobar (it's ∇H·f_slaved, not the variation of
   H *along* the isobar, which is identically zero);
4. advance `X ← X - 2π` and repeat.

All integrals are frozen-`c` 1-D trapezoidal quadratures; no stiff ODE. Freezing `c`
within a period is consistent to `O((St ε²)²)`. Returns `(; X, z, t, x, c)` at every
quadrature node, `x = X + t`. `c` is piecewise-constant per period (it jumps at the seam).
"""
function slaved_isobar_xz(p::Params; X0=0.0, z0=0.0,
                          nperiods=400, nquad=200, zmin=-8.0)
    H    = hamiltonian(p)
    Xdot = xdot_slaved(p)
    Hdot = hdot_drift(p)
    zof  = invert_isobar(p)
    h    = 2π / nquad                      # |ΔX| per node (X decreases along the path)

    Xs = Float64[]; zs = Float64[]; ts = Float64[]; xs = Float64[]; cs = Float64[]

    Xk = X0
    c  = H(X0, z0)
    t  = 0.0
    zseed = z0
    push!(Xs, X0); push!(zs, z0); push!(ts, t); push!(xs, X0 + t); push!(cs, c)

    for _ in 1:nperiods
        Δc = 0.0
        # left endpoint of this period (integrand only; node already stored)
        zj         = zof(Xk, c, zseed)
        invXd_prev = 1.0 / Xdot(Xk, zj)
        dc_prev    = Hdot(Xk, zj) * invXd_prev
        for j in 1:nquad
            Xj1   = Xk - j*h
            zj1   = zof(Xj1, c, zj)        # Newton seeded from previous node ⇒ physical branch
            invXd = 1.0 / Xdot(Xj1, zj1)
            dc    = Hdot(Xj1, zj1) * invXd
            t  += (-h) * 0.5 * (invXd_prev + invXd)   # ∫ dX/Ẋ,  dX = -h
            Δc += (-h) * 0.5 * (dc_prev + dc)         # ∫ (Ḣ/Ẋ) dX
            push!(Xs, Xj1); push!(zs, zj1); push!(ts, t); push!(xs, Xj1 + t); push!(cs, c)
            zj = zj1; invXd_prev = invXd; dc_prev = dc
            zj1 < zmin && return (; X=Xs, z=zs, t=ts, x=xs, c=cs)
        end
        c    += Δc                          # the ONE label update per wave period
        Xk   -= 2π
        zseed = zj
    end
    return (; X=Xs, z=zs, t=ts, x=xs, c=cs)
end

"""
    slaved_flow(p::Params) -> Function
Full slaved wave-frame field `(Ẋ, ż)`,
`Ẋ = -1 + eᶻ Re(A e^{iX}) + Sᵢ e^{2z}`, `ż = -vz + eᶻ Im(A e^{iX}) - Sᵣ e^{2z}`,
i.e. the `B₁,₂=0` flow plus the drift `e^{2z}(Sᵢ, -Sᵣ)`. Returns `(X,z) -> (Ẋ, ż)`.
"""
function slaved_flow(p::Params)
    A = amplitude(p); S = slaved_amplitude(p); vz = p.vz
    Sr = real(S); Si = imag(S)
    return function (X, z)
        e = exp(z); e2 = exp(2z); w = A*exp(im*X)
        return (-1 + e*real(w) + Si*e2, -vz + e*imag(w) - Sr*e2)
    end
end

"""
    integrate_slaved(p::Params; X0, z0, dt, tmax, zmin) -> (; X, z, t, x)
Direct RK4 integration of [`slaved_flow`](@ref) — ground truth for the plotter.
"""
function integrate_slaved(p::Params; X0=0.0, z0=0.0, dt=1e-3, tmax=400.0, zmin=-8.0)
    f = slaved_flow(p)
    X = X0; z = z0; t = 0.0
    Xs = [X]; zs = [z]; ts = [t]; xs = [X + t]
    while t < tmax && z > zmin
        k1 = f(X, z)
        k2 = f(X + 0.5*dt*k1[1], z + 0.5*dt*k1[2])
        k3 = f(X + 0.5*dt*k2[1], z + 0.5*dt*k2[2])
        k4 = f(X + dt*k3[1],     z + dt*k3[2])
        X += dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6
        z += dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6
        t += dt
        push!(Xs, X); push!(zs, z); push!(ts, t); push!(xs, X + t)
    end
    return (; X=Xs, z=zs, t=ts, x=xs)
end

"""
    plot_slaved_isobar(p::Params; kwargs...) -> Plots.Plot
Three panels: (1) physical `(x,z)` path, per-period reconstruction vs. slaved ODE truth;
(2) wave-frame `(X,z)` with H-isobars in grey and the path crossing them (the visual
proof that `Ḣ ≠ 0`); (3) the label `c = H` climbing — the staircase is the per-period
update, overlaid on `H` along the true path. Logs the final advection distance from both.
"""
function plot_slaved_isobar(p::Params; X0=0.0, z0=0.0,
                            nperiods=200, nquad=200, zmin=-6.0)
    H   = hamiltonian(p)
    A = amplitude(p)
    T = 5/p.vz

    # full Maxey-Riley, released exactly on the slow manifold: v = A·E - i·vz at (X0,z0)
    E0 = exp(z0 + im*X0)
    u0 = [X0, z0, real(A*E0), imag(A*E0) - p.vz]
    # approximation
    t0 = time()
    rec = slaved_isobar_xz(p; X0=X0, z0=z0, nperiods=nperiods, nquad=nquad, zmin=zmin)
    t1 = time()
    # integrating the 2D system
    gt  = integrate_slaved(p; X0=X0, z0=z0, dt=1e-3, zmin=zmin)
    t2 = time()
    mr = solve(ODEProblem(maxey_riley!, u0, (0.0, T), p),
               abstol=1e-10, reltol=1e-10, dtmax=0.01, maxiters=1e9)
    t3 = time()
    println("Time to integrate the 2D system: $(t2-t1)")
    println("Time to integrate the iterative approximation: $(t1-t0)")
    println("Time to integrate the 4D system: $(t3-t2)")
    p1 = plot(gt.x, gt.z; label="slaved ODE", lw=2, color=:black,
              xlabel="x", ylabel="z", title="trajectory")
    plot!(p1, rec.x, rec.z; label="per-period isobar", lw=2, ls=:dash, color=:crimson)
    plot!(p1, [mr(s)[1] for s in mr.t], [mr(s)[2] for s in mr.t],
          label="Maxey-Riley (released on manifold)", lw=3, color=:orange, alpha=0.6)

    Xg = range(minimum(rec.X), maximum(rec.X); length=220)
    zg = range(minimum(rec.z) - 0.2, 0.2; length=220)
    p2 = contour(Xg, zg, (X, z) -> H(X, z); levels=25, color=:grays, colorbar=false,
                 xlabel="X = x - t", ylabel="z", title="isobars (grey) & path")
    plot!(p2, rec.X, rec.z; lw=2, color=:crimson, label="path (??? looks like there is a mistake)")

    p3 = plot(rec.X, rec.c; lw=2, color=:crimson, label="c (per-period update)",
              xlabel="X = x - t", ylabel="H", title="label drift  ∮ Ḣ/Ẋ dX ≠ 0")
    plot!(p3, gt.X, [H(X, z) for (X, z) in zip(gt.X, gt.z)];
          lw=2, ls=:dash, color=:black, label="H along slaved path")

    @info "final advection distance" reconstruction=rec.x[end] truth=gt.x[end]

    plt = plot(p1, p2, p3; layout=(1, 3), size=(1350, 420), legend=:best)
    savefig(plt, "figures/slaved.svg")
    return plt
end
