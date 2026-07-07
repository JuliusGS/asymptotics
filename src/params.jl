"""
    Params(; St, eps, b, vz)

Physics parameters for the decoupled Maxey–Riley / MSA problem, in the **`v_z`
convention**: `vz` is the terminal settling velocity directly (NOT `St*θ`). The
Maxey–Riley gravity term is `-vz/St`, so a particle settles at exactly `vz` in still
fluid, independent of `St`.

Immutable and physics-only. Integration details (initial condition, `tspan`, which
det-function, callbacks) are passed explicitly to the solver helpers, never stored here.

* `St`  — Stokes number
* `eps` — wave slope ε (steepness `a*k`)
* `b`   — added-mass coefficient (`b = 1` ⇒ neutrally buoyant, response amplitude `A = ε`)
* `vz`  — terminal settling velocity
"""
Base.@kwdef struct Params
    St::Float64
    eps::Float64
    b::Float64
    vz::Float64
end
