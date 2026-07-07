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
