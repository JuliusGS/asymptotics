# Solver utilities vendored from the 2026.03 `plot_asymptotics.jl`, adapted so the
# integration details (det-function, initial condition, tspan, callbacks) are explicit
# arguments instead of fields read off a mutable params struct.

"""
    ode_path_gen(det_function, u0, tspan, p::Params;
                 K=Inf, callback_bottom="none", callback_top="none",
                 dtmax=0.005, maxiters=1e9) -> ODESolution

Solve `ODEProblem(det_function, u0, tspan, p)` with optional finite-depth (`K`)
boundary callbacks. For the decoupled deep-water case use the defaults (`K = Inf`, no
callbacks). The free-surface "pushdown" top callbacks of the original are omitted (they
are unused in the decoupled setting).
"""
function ode_path_gen(det_function, u0, tspan, p::Params;
                      K=Inf, callback_bottom="none", callback_top="none",
                      dtmax=0.005, maxiters=1e9)
    condition_fd(u, t, integrator) = u[2] + K
    condition_top(u, t, integrator) = u[2]
    function push_up_2d!(integrator)
        integrator.u[2] = -K + 0.1
        integrator.u[4] = -integrator.u[4]
    end

    if callback_bottom == "pure absorption"
        bottom = ContinuousCallback(condition_fd, terminate!, save_positions=(true, false))
    elseif callback_bottom == "no-flux"
        bottom = ContinuousCallback(condition_fd, push_up_2d!, save_positions=(true, false))
    elseif callback_bottom == "none"
        bottom = nothing
    else
        error("Unknown callback_bottom: $callback_bottom")
    end

    if callback_top == "absorption"
        top = ContinuousCallback(condition_top, terminate!, save_positions=(true, false))
    elseif callback_top == "none"
        top = nothing
    else
        error("Unknown callback_top: $callback_top")
    end

    prob = ODEProblem(det_function, u0, tspan, p)
    return solve(prob, callback=CallbackSet(bottom, top), dtmax=dtmax, maxiters=maxiters)
end

"""
    path_distance_var(p::Params, det_full, u0_full, det_approx, u0_approx, tspan;
                      da_length=1.5, ga_param=3) -> Float64

Double-averaged, `FAD²`-normalised relative error between a full trajectory
(`det_full`) and an approximation (`det_approx`), both solved with [`ode_path_gen`](@ref).
This is the paper's grading metric; only the double-average branch is vendored (the
original's `make_plots=false, do_gaussian=false` path), so none of the presentation
globals of `plot_asymptotics.jl` are needed. `FAD` is the full path's advection distance.
"""
function path_distance_var(p::Params, det_full, u0_full, det_approx, u0_approx, tspan;
                           da_length=1.5, ga_param=3)
    path  = ode_path_gen(det_full,   u0_full,   tspan, p)
    asymp = ode_path_gen(det_approx, u0_approx, tspan, p)

    # nearest time at which the wave-frame position x-t has advanced by one period (2π)
    make_np(pat) = function (t)
        f(tau) = pat(t)[1] - t - (pat(tau)[1] - tau) - 2pi
        for k in 1:10
            a, b = t + k*pi, t + (k+1)*pi
            f(a)*f(b) < 0 && return find_zero(f, (a, b))
        end
        error("No sign change for t=$t within 10pi")
    end
    np_path  = make_np(path)
    np_asymp = make_np(asymp)

    function lag_avg(pat, np_fn, t)
        (t < 0 || t > last(pat.t) - 10pi) && return [NaN, NaN]
        [mean(pat(tau)[1] for tau in t:0.1:np_fn(t)),
         mean(pat(tau)[2] for tau in t:0.1:np_fn(t))]
    end
    function double_avg(pat, np_fn, t)
        (t < da_length/2 || t > last(pat.t) - 10pi - da_length/2) && return [NaN, NaN]
        [mean(lag_avg(pat, np_fn, tau)[1] for tau in (t-da_length/2):0.1:(t+da_length/2)),
         mean(lag_avg(pat, np_fn, tau)[2] for tau in (t-da_length/2):0.1:(t+da_length/2))]
    end

    tmax = min(last(path.t), last(asymp.t)) - 10pi - da_length/2 - 0.1
    tl   = range(da_length/2 + 0.1, tmax, 50)
    FAD  = sqrt((path(last(path.t))[1] - path(0)[1])^2 +
                (path(last(path.t))[2] - path(0)[2])^2)
    err = mean((double_avg(asymp, np_asymp, t)[1] - double_avg(path, np_path, t)[1])^2 +
               (double_avg(asymp, np_asymp, t)[2] - double_avg(path, np_path, t)[2])^2
               for t in tl) / FAD^2
    isnan(err) && error("Double-average error is NaN (integration window too short)")
    return err
end

# --- (σ, ω) machinery: initial condition, solve, collapse, and lab-frame recovery ---

"""
    sigma_omega_ic(p::Params, u0) -> Vector{ComplexF64}

Initial condition `[σ₀, σ̄₀, ω₀, ω̄₀]` for [`sigma_omega!`](@ref) from a lab-frame
state `u0 = [x, z, vx, vz]`.
"""
function sigma_omega_ic(p::Params, u0)
    eps = p.eps; b = p.b
    s0 = exp(u0[2] - im*u0[1])
    o0 = (u0[3] - 1 - b*eps*exp(u0[2])*cos(u0[1])) +
         im*(u0[4] - b*eps*exp(u0[2])*sin(u0[1]))
    return ComplexF64[s0, conj(s0), o0, conj(o0)]
end

"""
    sigma_omega_solution(p::Params, u0, tspan; solve_kwargs...) -> (; sol, sigma, omega)

Solve the reduced `(σ, ω)` system from lab-frame IC `u0` over `tspan`. Returns the
`ODESolution` plus scalar interpolants `sigma(t) = sol(t)[1]`, `omega(t) = sol(t)[3]`.
"""
function sigma_omega_solution(p::Params, u0, tspan; solve_kwargs...)
    prob = ODEProblem(sigma_omega!, sigma_omega_ic(p, u0), tspan, p; maxiters=1e9)
    sol  = solve(prob; solve_kwargs...)
    return (; sol, sigma = t -> sol(t)[1], omega = t -> sol(t)[3])
end

"""
    collapse_sigma_omega(p::Params, sigma, omega) -> (sigma_simpl, omega_simpl)

Project the time-dependent `(σ, ω)` interpolants onto the slow eigenspace (subtract the
fixed point, apply [`projection`](@ref), add it back). Returns closures of `t`.
"""
function collapse_sigma_omega(p::Params, sigma, omega)
    P = projection(p); vz = p.vz
    fp = ComplexF64[0, 0, -1 - im*vz, -1 + im*vz]
    simpl(t) = P*(ComplexF64[sigma(t), conj(sigma(t)), omega(t), conj(omega(t))] - fp) + fp
    return (t -> simpl(t)[1], t -> simpl(t)[3])
end

"""
    recover_xz_from_sigma_omega(p::Params, sigma, omega, u0_xz, tspan; solve_kwargs...)
        -> ODESolution

Integrate the lab-frame flow `d(x+iz)/dt = 1 + ω(t) + b ε conj(σ(t))` driven by the
given `(σ, ω)` interpolants (passed by closure — no globals). `u0_xz = [x0, z0]`.
"""
function recover_xz_from_sigma_omega(p::Params, sigma, omega, u0_xz, tspan; solve_kwargs...)
    b = p.b; eps = p.eps
    function det!(dw, w, _p, t)
        v = 1 + omega(t) + b*eps*conj(sigma(t))
        dw[1] = real(v)
        dw[2] = imag(v)
    end
    return solve(ODEProblem(det!, u0_xz, tspan, p; maxiters=1e9); solve_kwargs...)
end
