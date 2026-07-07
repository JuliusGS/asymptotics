# Accuracy vs St: the slow-manifold approximation and the fluid+settling reference,
# each graded against the full Maxey-Riley solution with the paper's double-averaging
# metric (path_distance_var). Successor to compare_st_accuracy in
# code/collapse_onto_eigenspace.jl.
#
# Coupled physics: θ = 1-b held fixed, so the terminal velocity vz = St(1-b) scales with
# St across the sweep (matching the old theta-convention sweep).
#
#   julia --project=. scripts/st_accuracy.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using MSA
using Plots, LaTeXStrings

function st_accuracy(; n_pts=10, St_lo=0.01, St_hi=1.0, tend=200.0, eps=0.33, b=0.9,
                     outdir=joinpath(@__DIR__, "..", "figures"))
    Sts = 10 .^ range(log10(St_lo), log10(St_hi), length=n_pts)
    e_slow  = fill(NaN, n_pts)
    e_fluid = fill(NaN, n_pts)
    u0_full = [0.0, 0.0, 0.33, 0.0]      # release at fluid velocity
    for (i, Sti) in enumerate(Sts)
        p = Params(St=Sti, eps=eps, b=b, vz=Sti*(1 - b))
        tspan = (0.0, tend)
        try
            e_slow[i]  = path_distance_var(p, maxey_riley!, u0_full, slow_manifold_xz!, [0.0, 0.0], tspan)
        catch err
            println("  slow  St=$(round(Sti,digits=4)) failed: $err")
        end
        try
            e_fluid[i] = path_distance_var(p, maxey_riley!, u0_full, fluid_settling!, [0.0, 0.0], tspan)
        catch err
            println("  fluid St=$(round(Sti,digits=4)) failed: $err")
        end
        println("St=$(round(Sti,digits=4))   slow=$(e_slow[i])   fluid=$(e_fluid[i])")
    end

    plot(Sts, e_slow; xscale=:log10, yscale=:log10, marker=:circle, lw=2,
         label="slow manifold (B₁,₂=0)", xlabel=L"\mathrm{St}",
         ylabel="rel. double-avg error", legend=:topleft)
    plot!(Sts, e_fluid; marker=:diamond, lw=2, ls=:dash, label="fluid + settling")
    mkpath(outdir)
    fn = joinpath(outdir, "st_accuracy_comparison_eps$(eps)_b$(b).svg")
    savefig(fn)
    println("saved ", fn)
    return Sts, e_slow, e_fluid
end

st_accuracy()
