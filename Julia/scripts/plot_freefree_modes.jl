using Surferbot
using LaTeXStrings
using Printf
import Plots

const FIG_DIR = joinpath(@__DIR__, "../output/figures")

function freefree_modes_l2(n_modes::Int; n_pts::Int = 600)
    xi = range(0.0, 1.0; length = n_pts)
    dxi = step(xi)
    modes = Vector{Vector{Float64}}(undef, n_modes)
    if n_modes >= 1
        w = ones(n_pts)
        modes[1] = w ./ sqrt(sum(w .^ 2) * dxi)
    end
    if n_modes >= 2
        w = collect(xi) .- 0.5
        modes[2] = w ./ sqrt(sum(w .^ 2) * dxi)
    end
    if n_modes >= 3
        betaL_roots = Surferbot.freefree_betaL_roots(n_modes - 2)
        for k in 1:(n_modes - 2)
            w = Surferbot.freefree_mode_shape(collect(xi), 1.0, betaL_roots[k])
            modes[2 + k] = w ./ sqrt(sum(w .^ 2) * dxi)
        end
    end
    return collect(xi) .- 0.5, modes
end

function plot_modes()
    Plots.gr()
    
    xi, modes = freefree_modes_l2(5)
    n_modes = length(modes)
    beta_vals = vcat(0.0, 0.0, Surferbot.freefree_betaL_roots(n_modes - 2))

    y_all = vcat(modes...)
    ylim_abs = max(abs(minimum(y_all)), abs(maximum(y_all))) * 1.35
    ylims = (-ylim_abs, ylim_abs)

    subplots = []
    for i in 1:n_modes
        n = i - 1
        is_bottom = (i == n_modes)
        
        p = Plots.plot(xi, modes[i];
            color = :black,
            linewidth = 1.8,
            label = false,
            xlims = (-0.5, 0.5),
            ylims = ylims,
            xticks = is_bottom ? (-0.5:0.25:0.5, [L"-0.5", L"-0.25", L"0", L"0.25", L"0.5"]) : false,
            yticks = ([-2.0, 0.0, 2.0], [L"-2", L"0", L"2"]),
            xlabel = is_bottom ? L"x" : "",
            ylabel = i == 3 ? L"W_n(x)" : "",
            grid = false,
            framestyle = :box,
            tick_direction = :out,
            fontfamily = "Computer Modern",
            guidefontsize = 19,
            tickfontsize = 13,
            left_margin = 12Plots.mm,
            right_margin = 8Plots.mm,
            top_margin = i == 1 ? 2Plots.mm : -4Plots.mm,
            bottom_margin = is_bottom ? 8Plots.mm : 0Plots.mm,
        )
        Plots.hline!(p, [0.0]; color = :black, alpha = 0.25, linewidth = 0.8, label = false)
        
        label_y = ylim_abs * 0.70
        label_str = @sprintf("\$n = %d,\\; \\beta_n = %.2f\$", n, beta_vals[i])
        Plots.annotate!(p, -0.48, label_y, Plots.text(label_str, :left, 13, "Computer Modern"))
        push!(subplots, p)
    end

    fig = Plots.plot(subplots...;
        layout = (n_modes, 1),
        size = (700, 540),
        dpi = 220,
        link = :x,
    )

    outfile = joinpath(FIG_DIR, "plot_freefree_modes")
    Plots.savefig(fig, outfile * ".pdf")
    Plots.savefig(fig, outfile * ".png")
    println("Saved $outfile.{pdf,png} (via Plots.jl)")
end

function main()
    mkpath(FIG_DIR)
    plot_modes()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
