# Figure-makers: one function per figure, all exported and callable directly after
# `using MSA` alone. These call `savefig`/`mkpath` (side effects when CALLED), but define
# nothing that runs when the module LOADS — that's the only "no side effects" rule.

"Filename suffix `St..._eps..._b..._vz...` encoding all run parameters."
figure_suffix(p::Params) = "St$(p.St)_eps$(p.eps)_b$(p.b)_vz$(round(p.vz, digits=4))"

const DEFAULT_FIGDIR = joinpath(@__DIR__, "..", "figures")

"""
    mr_fs_isobar_figure(p::Params; N=3000, tend=5/p.vz, prefix="mr_fs_isobar", outdir=...)

Lab-frame trajectory comparison in the decoupled setting, all released from
`(x,z) = (0,0)`: the isobar of the slow-manifold Hamiltonian, full Maxey-Riley from
rest, full Maxey-Riley from fluid velocity, and fluid+settling. Saves an SVG into
`outdir` named with all of `p`'s parameters.
"""
function mr_fs_isobar_figure(p::Params; N=3000, tend=nothing, prefix="mr_fs_isobar",
                             outdir=DEFAULT_FIGDIR)
    T = tend === nothing ? 5/p.vz : tend

    sol4(u0) = solve(ODEProblem(maxey_riley!, u0, (0.0, T), p),
                     abstol=1e-9, reltol=1e-9, saveat=0.05, dense=false, maxiters=1e8)
    mr_rest  = sol4([0.0, 0.0, 0.0, 0.0])       # M-R from rest
    mr_fluid = sol4([0.0, 0.0, p.eps, 0.0])     # M-R at fluid velocity (ε u at the origin)
    fs = solve(ODEProblem(fluid_settling!, [0.0, 0.0], (0.0, T), p),
               abstol=1e-9, reltol=1e-9, saveat=0.05, dense=false)

    rec = recover_xzt(p; X0=0.0, z0=0.0, Xspan=T, N=N)   # isobar -> (x, z)

    plt = plot(rec.x, rec.z, label="isobar (simplified Hamiltonian)", lw=2.5, color=:blue,
               xlabel="x", ylabel="z", size=(1000, 650), legend=:bottomright)
    plot!(plt, [u[1] for u in mr_rest.u],  [u[2] for u in mr_rest.u],
          label="M-R (from rest)", lw=2, color=:red)
    plot!(plt, [u[1] for u in mr_fluid.u], [u[2] for u in mr_fluid.u],
          label="M-R (from fluid velocity)", lw=2, color=:seagreen)
    plot!(plt, [u[1] for u in fs.u], [u[2] for u in fs.u],
          label="fluid + settling", lw=2, ls=:dash, color=:purple)

    mkpath(outdir)
    fn = joinpath(outdir, "$(prefix)_$(figure_suffix(p)).svg")
    savefig(plt, fn)
    println("saved ", fn, "   A = ", round(amplitude(p), digits=5))
    return plt
end

"""
    compare_full_fs_ham(p::Params; X0=0.0, z0=0.0, N=5000, outdir=...)

Full Maxey-Riley (released ON the slow manifold) vs. the isobar reconstruction and
fluid+settling. All three use a single, consistent `vz`, so isobar and full M-R agree
to solver tolerance (unlike the old `code/fixed_xzt.jl`, which mixed conventions).
"""
function compare_full_fs_ham(p::Params; X0=0.0, z0=0.0, N=5000, outdir=DEFAULT_FIGDIR)
    A = amplitude(p)
    T = 5/p.vz

    # full Maxey-Riley, released exactly on the slow manifold: v = A·E - i·vz at (X0,z0)
    E0 = exp(z0 + im*X0)
    u0 = [X0, z0, real(A*E0), imag(A*E0) - p.vz]
    mr = solve(ODEProblem(maxey_riley!, u0, (0.0, T), p),
               abstol=1e-10, reltol=1e-10, dtmax=0.01, maxiters=1e9)

    fs = solve(ODEProblem(fluid_settling!, [X0, z0], (0.0, T), p),
               abstol=1e-10, reltol=1e-10, dtmax=0.01)

    rec = recover_xzt(p; X0=X0, z0=z0, Xspan=T, N=N)

    plt = plot(rec.x, rec.z, label="isobar reconstruction", lw=2, color=:blue,
               xlabel=L"x", ylabel=L"z", size=(1000, 640),
               title="St=$(p.St), b=$(p.b), vz=$(p.vz)")
    plot!(plt, [mr(s)[1] for s in mr.t], [mr(s)[2] for s in mr.t],
          label="Maxey-Riley (released on manifold)", lw=3, color=:orange, alpha=0.6)
    plot!(plt, [fs(s)[1] for s in fs.t], [fs(s)[2] for s in fs.t],
          label="fluid + settling", lw=2, ls=:dash, color=:purple)

    mkpath(outdir)
    fn = joinpath(outdir, "compare_full_fs_ham_$(figure_suffix(p)).svg")
    savefig(plt, fn)
    println("endpoints:  isobar x=$(round(rec.x[end],digits=4)) z=$(round(rec.z[end],digits=3)) | ",
            "MR x=$(round(mr(T)[1],digits=4)) z=$(round(mr(T)[2],digits=3)) | ",
            "fs x=$(round(fs(T)[1],digits=4)) z=$(round(fs(T)[2],digits=3))")
    println("saved ", fn)
    return plt
end

"""
    b3_contours(p::Params; n=100, outdir=...)

`H̄(B₃)` over the complex `B₃`-plane — Cartesian and polar grids, real and imaginary
parts (4 SVGs).
"""
function b3_contours(p::Params; n=100, outdir=DEFAULT_FIGDIR)
    Hbar = hamiltonian_B3(p)
    mkpath(outdir)
    x = range(-10, 10, length=n); y = range(-10, 10, length=n)
    for (part, f) in (("real", real), ("imag", imag))
        Z = @. f(Hbar(x' + im*y))
        contour(x, y, Z, fill=true, xlabel=L"\Re(B_3)", ylabel=L"\Im(B_3)",
                colorbar_title=part == "real" ? L"\Re(H)" : L"\Im(H)")
        savefig(joinpath(outdir, "contour_of_$(part)_hamiltonian_$(figure_suffix(p)).svg"))
    end
    r = range(0, 10, length=n); ang = range(-pi, pi, length=n)
    for (part, f) in (("real", real), ("imag", imag))
        Z = @. f(Hbar(r' * exp(im*ang)))
        contour(r, ang, Z, fill=true, xlabel=L"r(B_3)", ylabel=L"\arg(B_3)",
                colorbar_title=part == "real" ? L"\Re(H)" : L"\Im(H)")
        savefig(joinpath(outdir, "polar_contour_of_$(part)_hamiltonian_$(figure_suffix(p)).svg"))
    end
end

"""
    xz_contours(p::Params; xlim=(0,20pi), zlim=(-5,2), n=500, outdir=...)

`H(X, z)` in wave-frame coordinates; its level curves are the isobars.
"""
function xz_contours(p::Params; xlim=(0, 20pi), zlim=(-5, 2), n=500, outdir=DEFAULT_FIGDIR)
    H = hamiltonian(p)
    X = range(xlim[1], xlim[2], length=n); z = range(zlim[1], zlim[2], length=n)
    Hgrid = [H(x, zz) for zz in z, x in X]
    contour(X, z, Hgrid, fill=true, levels=30, xlabel=L"X = x-t", ylabel=L"z",
            colorbar_title=L"H(X,z)")
    mkpath(outdir)
    savefig(joinpath(outdir, "hamiltonian_XZ_$(figure_suffix(p)).svg"))
end

"""
    sigma_omega_contours(p::Params; lim=500.0, n=100, outdir=...)

`H(σ, ω)` via the collapse-onto-slow-eigenspace map (2 SVGs).
"""
function sigma_omega_contours(p::Params; lim=500.0, n=100, outdir=DEFAULT_FIGDIR)
    Ht = ham_trafo(p)
    mkpath(outdir)
    s = range(-lim, lim, length=n); o = range(-lim, lim, length=n)
    for (part, f) in (("real", real), ("imag", imag))
        Z = @. f(Ht(s', o))
        contour(s, o, Z, fill=true, xlabel=L"\sigma", ylabel=L"\omega",
                colorbar_title=part == "real" ? L"\Re(H)" : L"\Im(H)")
        savefig(joinpath(outdir, "sigma_omega_contour_of_$(part)_hamiltonian_$(figure_suffix(p)).svg"))
    end
end

"""
    sigma_omega_recovery_figure(p::Params; u0=[0,0,0.33,0], tspan=(0,200), outdir=...)

Recover the lab-frame trajectory from the reduced `(σ, ω)` system, and overlay the
recovery after collapsing `(σ, ω)` onto the 2-D slow eigenspace — the two should barely
differ (`B₁, B₂ ≈ 0`).
"""
function sigma_omega_recovery_figure(p::Params; u0=[0.0, 0.0, 0.33, 0.0], tspan=(0.0, 200.0),
                                     outdir=DEFAULT_FIGDIR)
    so = sigma_omega_solution(p, u0, tspan)
    full = recover_xz_from_sigma_omega(p, so.sigma, so.omega, u0[1:2], tspan)

    sig_s, om_s = collapse_sigma_omega(p, so.sigma, so.omega)
    simpl = recover_xz_from_sigma_omega(p, sig_s, om_s, u0[1:2], tspan)

    plt = plot([full(t)[1] for t in full.t], [full(t)[2] for t in full.t],
               label="full (σ,ω) solution", lw=2, xlabel=L"x", ylabel=L"z", size=(1000, 640),
               title="St=$(p.St), b=$(p.b), vz=$(round(p.vz,digits=4))")
    plot!(plt, [simpl(t)[1] for t in simpl.t], [simpl(t)[2] for t in simpl.t],
          label="recovered after 2-D eigenspace projection", lw=2, ls=:dash)

    mkpath(outdir)
    fn = joinpath(outdir, "sigma_omega_recovery_$(figure_suffix(p)).svg")
    savefig(plt, fn)
    ts = range(tspan[1], min(last(full.t), last(simpl.t)), length=500)
    dmax = maximum(hypot(full(t)[1]-simpl(t)[1], full(t)[2]-simpl(t)[2]) for t in ts)
    println("max |full − projected| over path = ", dmax)
    println("saved ", fn)
    return plt
end

"""
    st_accuracy_figure(; n_pts=10, St_lo=0.01, St_hi=1.0, tend=200.0, eps=0.33, b=0.9, outdir=...)

Accuracy vs `St`: grade the slow-manifold approximation and the fluid+settling
reference against full Maxey-Riley with the double-averaging metric
([`path_distance_var`](@ref)). `θ = 1-b` held fixed, so `vz = St(1-b)` scales with `St`.
"""
function st_accuracy_figure(; n_pts=10, St_lo=0.01, St_hi=1.0, tend=200.0, eps=0.33, b=0.9,
                            outdir=DEFAULT_FIGDIR)
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

"""
    final_advection_vs_st_figure(; St_lo=0.01, St_hi=1.0, n=200, eps=0.33, b=0.5, vz=0.05,
                                 outdir=DEFAULT_FIGDIR) -> (Sts, d)

The slow-manifold model's final advection distance ([`final_advection_distance`](@ref))
as a function of Stokes number, with `eps`, `b`, `vz` held fixed (St varied *only*). `St`
is log-spaced over `[St_lo, St_hi]`. The dashed line is the `b = 1` inertialess baseline
(`A = ε`), so the gap between the curves is the inertial correction at each St.
"""
function final_advection_vs_st_figure(; St_lo=0.01, St_hi=1.0, n=200, eps=0.33, b=0.5, vz=0.05,
                                      outdir=DEFAULT_FIGDIR)
    Sts = 10 .^ range(log10(St_lo), log10(St_hi), length=n)
    d      = [final_advection_distance(Params(St=St, eps=eps, b=b,   vz=vz)) for St in Sts]
    d_iner = [final_advection_distance(Params(St=St, eps=eps, b=1.0, vz=vz)) for St in Sts]

    plot(Sts, d; xscale=:log10, lw=2.5, color=:blue, label="b = $b",
         xlabel=L"\mathrm{St}", ylabel="final advection distance",
         title="ε=$eps, b=$b, vz=$vz", legend=:best)
    plot!(Sts, d_iner; lw=2, ls=:dash, color=:black, label="b = 1 (inertialess)")

    mkpath(outdir)
    fn = joinpath(outdir, "final_advection_vs_St_eps$(eps)_b$(b)_vz$(round(vz, digits=4)).svg")
    savefig(fn)
    println("saved ", fn)
    return Sts, d
end

"""
    final_advection_correction_order_figure(; St_lo=1e-3, St_hi=0.3, n=150,
        eps=0.33, b=0.5, vz=0.05, full_mr=true, n_mr=14, outdir=DEFAULT_FIGDIR)

Order-in-St diagnostic for the inertial correction `Δ(St) = |d(St; b) − d_fs|` of the
final advection distance, where `d_fs` is the **St = 0 limit** — the fluid+settling
distance ([`final_advection_distance_fs`](@ref)), a single common baseline for both
curves (it coincides with the `b = 1` distance in both descriptions). Two curves, both in
the decoupled (`vz`, `St` independent) convention:

* **slow-manifold approximation** ([`final_advection_distance`](@ref)) — cheap, fine grid;
* **full Maxey–Riley** ([`final_advection_distance_mr`](@ref)) — ODE solves, coarse grid
  (`n_mr` points, `St ≥ 0.01`); set `full_mr=false` to skip.

Log–log, with dashed guides `∝ St` and `∝ St²` anchored at the small-St end. Both curves
track `∝ St` (the correction is O(St), carried by the O(St) phase lag), though their
coefficients differ — the reduction gets the order right, not the coefficient.
"""
function final_advection_correction_order_figure(; St_lo=1e-3, St_hi=0.3, n=150,
        eps=0.33, b=0.5, vz=0.05, full_mr=true, n_mr=14, outdir=DEFAULT_FIGDIR)
    d_fs = final_advection_distance_fs(Params(St=1.0, eps=eps, b=b, vz=vz))  # St=0 limit (indep. of St,b)

    # slow-manifold approximation (cheap: fine grid), vs the fluid+settling St=0 baseline
    Sts = 10 .^ range(log10(St_lo), log10(St_hi), length=n)
    aΔ  = [abs(final_advection_distance(Params(St=St, eps=eps, b=b, vz=vz)) - d_fs) for St in Sts]

    plt = plot(Sts, aΔ; xscale=:log10, yscale=:log10, lw=2.5, color=:blue,
               label="slow-manifold approximation", xlabel=L"\mathrm{St}",
               ylabel=L"|d(\mathrm{St};\,b) - d_{\mathrm{fs}}|",
               title="ε=$eps, b=$b, vz=$vz", legend=:bottomright)

    # full Maxey–Riley (expensive ODE solves: coarse grid, St ≥ 0.01), same baseline
    Sts_mr = aΔ_mr = nothing
    if full_mr
        Sts_mr = 10 .^ range(log10(max(St_lo, 1e-2)), log10(St_hi), length=n_mr)
        aΔ_mr  = [abs(final_advection_distance_mr(Params(St=St, eps=eps, b=b, vz=vz)) - d_fs) for St in Sts_mr]
        plot!(plt, Sts_mr, aΔ_mr; seriestype=:scatter, marker=:circle, ms=5, color=:red,
              label="full Maxey–Riley")
    end

    # slope guides anchored at the smallest-St approximation point
    plot!(plt, Sts, aΔ[1] .* (Sts ./ Sts[1]);    ls=:dash, color=:gray,  label=L"\propto \mathrm{St}")
    plot!(plt, Sts, aΔ[1] .* (Sts ./ Sts[1]).^2; ls=:dot,  color=:black, label=L"\propto \mathrm{St}^2")

    mkpath(outdir)
    fn = joinpath(outdir, "final_advection_correction_order_eps$(eps)_b$(b)_vz$(round(vz, digits=4)).svg")
    savefig(plt, fn)
    println("saved ", fn)
    return (; Sts, aΔ, Sts_mr, aΔ_mr)
end

"""
    final_advection_avgstart_vs_st_figure(; St_lo=0.01, St_hi=1.0, n=20,
        eps=0.33, b=0.5, vz=0.05, outdir=DEFAULT_FIGDIR) -> (; Sts, d_avg, d_inst)

Full Maxey–Riley final advection distance vs St, measured from the **phase-averaged
start** ([`period_averaged_start`](@ref) / [`final_advection_distance_mr_avgstart`](@ref))
— i.e. averaging the position over the first oscillation period (delimited by negative→
positive `vₓ` crossings). The dashed curve is the same distance measured from the
instantaneous release point, for comparison. Particle released at fluid velocity.
"""
function final_advection_avgstart_vs_st_figure(; St_lo=0.01, St_hi=1.0, n=20,
        eps=0.33, b=0.5, vz=0.05, outdir=DEFAULT_FIGDIR)
    Sts = 10 .^ range(log10(St_lo), log10(St_hi), length=n)
    d_avg  = Float64[]
    d_inst = Float64[]
    for St in Sts
        p = Params(St=St, eps=eps, b=b, vz=vz)
        s = period_averaged_start(p)
        sol = solve(ODEProblem(maxey_riley!, [0.0, 0.0, eps, 0.0], (0.0, 30/vz), p),
                    abstol=1e-12, reltol=1e-10, save_everystep=false, maxiters=1e9)
        xend = sol.u[end][1]
        push!(d_avg,  xend - s.avgd_x0)
        push!(d_inst, xend)                              # instantaneous start (x_release = 0)
    end

    plot(Sts, d_avg; xscale=:log10, lw=2.5, marker=:circle, ms=4, color=:blue,
         label="from phase-averaged start", xlabel=L"\mathrm{St}",
         ylabel="final advection distance", title="ε=$eps, b=$b, vz=$vz", legend=:best)
    plot!(Sts, d_inst; lw=2, ls=:dash, marker=:diamond, ms=3, color=:red,
          label="from instantaneous release")

    mkpath(outdir)
    fn = joinpath(outdir, "final_advection_avgstart_vs_St_eps$(eps)_b$(b)_vz$(round(vz, digits=4)).svg")
    savefig(fn)
    println("saved ", fn)
    return (; Sts, d_avg, d_inst)
end

"""
    final_advection_avgstart_order_figure(; St_lo=1e-2, St_hi=0.3, n=16,
        eps=0.33, b=0.5, vz=0.05, outdir=DEFAULT_FIGDIR) -> (; Sts, aΔ)

Log–log order-in-St diagnostic for the full-M–R final advection distance measured from
the **phase-averaged start** ([`final_advection_distance_mr_avgstart`](@ref)), against its
St = 0 limit ([`final_advection_distance_fs_avgstart`](@ref)). Guides `∝ St` and `∝ St²`.
Tests whether averaging out the first-period oscillation removes the O(St) boundary term.
"""
function final_advection_avgstart_order_figure(; St_lo=1e-2, St_hi=0.3, n=16,
        eps=0.33, b=0.5, vz=0.05, outdir=DEFAULT_FIGDIR)
    Sts = 10 .^ range(log10(St_lo), log10(St_hi), length=n)
    d0  = final_advection_distance_fs_avgstart(Params(St=0.1, eps=eps, b=b, vz=vz))  # St=0 limit
    aΔ  = [abs(final_advection_distance_mr_avgstart(Params(St=St, eps=eps, b=b, vz=vz)) - d0) for St in Sts]

    plt = plot(Sts, aΔ; xscale=:log10, yscale=:log10, seriestype=:scatter, marker=:circle,
               ms=5, color=:blue, label="M–R, phase-averaged start", xlabel=L"\mathrm{St}",
               ylabel=L"|d_{\mathrm{avg}}(\mathrm{St}) - d_{\mathrm{avg}}(0)|",
               title="ε=$eps, b=$b, vz=$vz", legend=:bottomright)
    plot!(plt, Sts, aΔ[1] .* (Sts ./ Sts[1]);    ls=:dash, color=:gray,  label=L"\propto \mathrm{St}")
    plot!(plt, Sts, aΔ[1] .* (Sts ./ Sts[1]).^2; ls=:dot,  color=:black, label=L"\propto \mathrm{St}^2")

    mkpath(outdir)
    fn = joinpath(outdir, "final_advection_avgstart_order_eps$(eps)_b$(b)_vz$(round(vz, digits=4)).svg")
    savefig(plt, fn)
    println("saved ", fn, "   d0=", round(d0, digits=6))
    return (; Sts, aΔ)
end

"""
    final_advection_ddt_order_figure(; St_lo=1e-2, St_hi=0.3, n=14,
        eps=0.33, b=0.5, vz=0.05, outdir=DEFAULT_FIGDIR) -> (; Sts, aΔ_ddt, aΔ_DDt)

Log–log order-in-St diagnostic of the final advection distance (phase-averaged start,
[`final_advection_distance_mr_avgstart`](@ref)) for **both** the particle-following
[`maxey_riley!`](@ref) and DiBenedetto's fluid-`Du/Dt` [`maxey_riley_DDt!`](@ref), against
the shared fluid+settling St=0 baseline. Tests whether the O(St) correction is sensitive
to the added-mass derivative choice (it is not — both track `∝ St`).
"""
function final_advection_ddt_order_figure(; St_lo=1e-2, St_hi=0.3, n=14,
        eps=0.33, b=0.5, vz=0.05, outdir=DEFAULT_FIGDIR)
    Sts = 10 .^ range(log10(St_lo), log10(St_hi), length=n)
    d0  = final_advection_distance_fs_avgstart(Params(St=0.1, eps=eps, b=b, vz=vz))
    aΔ_ddt = [abs(final_advection_distance_mr_avgstart(Params(St=St,eps=eps,b=b,vz=vz);
                  dynamics=maxey_riley!)    - d0) for St in Sts]
    aΔ_DDt = [abs(final_advection_distance_mr_avgstart(Params(St=St,eps=eps,b=b,vz=vz);
                  dynamics=maxey_riley_DDt!) - d0) for St in Sts]

    plt = plot(Sts, aΔ_ddt; xscale=:log10, yscale=:log10, seriestype=:scatter, marker=:circle,
               ms=5, color=:blue, label="M–R, du/dt (particle)", xlabel=L"\mathrm{St}",
               ylabel=L"|d_{\mathrm{avg}}(\mathrm{St}) - d_{\mathrm{avg}}(0)|",
               title="ε=$eps, b=$b, vz=$vz", legend=:bottomright)
    plot!(plt, Sts, aΔ_DDt; seriestype=:scatter, marker=:diamond, ms=5, color=:red,
          label="M–R, Du/Dt (fluid, DiBenedetto)")
    plot!(plt, Sts, aΔ_ddt[1] .* (Sts ./ Sts[1]);    ls=:dash, color=:gray,  label=L"\propto \mathrm{St}")
    plot!(plt, Sts, aΔ_ddt[1] .* (Sts ./ Sts[1]).^2; ls=:dot,  color=:black, label=L"\propto \mathrm{St}^2")

    mkpath(outdir)
    fn = joinpath(outdir, "final_advection_ddt_order_eps$(eps)_b$(b)_vz$(round(vz, digits=4)).svg")
    savefig(plt, fn)
    println("saved ", fn)
    return (; Sts, aΔ_ddt, aΔ_DDt)
end

# Canonical parameter sets that reproduce the paper figures. This is the single record of
# "which p makes which figure" (it used to live one-per-file in scripts/).
"""
    make_all_figures(; outdir=DEFAULT_FIGDIR, include_st_sweep=true)

Regenerate every figure with its canonical parameter set — one call rebuilds `figures/`.
`include_st_sweep=false` skips the slow `st_accuracy_figure` sweep.

From the shell: `julia --project=. -e 'using MSA; make_all_figures()'`.
"""
function make_all_figures(; outdir=DEFAULT_FIGDIR, include_st_sweep=true)
    mr_fs_isobar_figure(Params(St=0.5, eps=0.33, b=0.5, vz=0.05); outdir)
    compare_full_fs_ham(Params(St=0.5, eps=0.33, b=0.9, vz=0.05); outdir)
    # coupled-physics set (θ = 1-b, so vz = St(1-b)) for the Hamiltonian / σ-ω figures
    pc = Params(St=0.8, eps=0.33, b=0.9, vz=0.8*(1 - 0.9))
    b3_contours(pc; outdir)
    xz_contours(pc; outdir)
    sigma_omega_contours(pc; outdir)
    sigma_omega_recovery_figure(pc; outdir)
    final_advection_vs_st_figure(; outdir)
    final_advection_correction_order_figure(; outdir, full_mr=include_st_sweep)
    include_st_sweep && st_accuracy_figure(; outdir)
    return nothing
end
