# Recover the lab-frame trajectory from the reduced (σ, ω) system, and check that
# collapsing (σ, ω) onto the 2-D slow eigenspace before recovery barely changes the path
# (i.e. B₁, B₂ are negligible). Successor to the main body of
# code/collapse_onto_eigenspace.jl.
#
#   julia --project=. scripts/sigma_omega_recovery.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using MSA
using DifferentialEquations, Plots, LaTeXStrings

function sigma_omega_recovery(p::Params; u0=[0.0, 0.0, 0.33, 0.0], tspan=(0.0, 200.0),
                              outdir=joinpath(@__DIR__, "..", "figures"))
    so = sigma_omega_solution(p, u0, tspan)

    # full recovery
    full = recover_xz_from_sigma_omega(p, so.sigma, so.omega, u0[1:2], tspan)

    # recovery after collapsing (σ, ω) onto the slow eigenspace
    sig_s, om_s = collapse_sigma_omega(p, so.sigma, so.omega)
    simpl = recover_xz_from_sigma_omega(p, sig_s, om_s, u0[1:2], tspan)

    plt = plot([full(t)[1] for t in full.t], [full(t)[2] for t in full.t],
               label="full (σ,ω) solution", lw=2, xlabel=L"x", ylabel=L"z", size=(1000, 640),
               title="St=$(p.St), b=$(p.b), vz=$(round(p.vz,digits=4))")
    plot!(plt, [simpl(t)[1] for t in simpl.t], [simpl(t)[2] for t in simpl.t],
          label="recovered after 2-D eigenspace projection", lw=2, ls=:dash)

    mkpath(outdir)
    fn = joinpath(outdir, "sigma_omega_recovery_St$(p.St)_eps$(p.eps)_b$(p.b)_vz$(round(p.vz,digits=4)).svg")
    savefig(plt, fn)
    # max discrepancy between full and projected recovery
    ts = range(tspan[1], min(last(full.t), last(simpl.t)), length=500)
    dmax = maximum(hypot(full(t)[1]-simpl(t)[1], full(t)[2]-simpl(t)[2]) for t in ts)
    println("max |full − projected| over path = ", dmax)
    println("saved ", fn)
    return plt
end

# Coupled-physics parameter set: θ = 1-b, vz = St(1-b).
sigma_omega_recovery(Params(St=0.8, eps=0.33, b=0.9, vz=0.8*(1 - 0.9)))
