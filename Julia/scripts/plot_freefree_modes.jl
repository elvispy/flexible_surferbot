using Surferbot
using CairoMakie
using LaTeXStrings
using Printf

include(joinpath(@__DIR__, "paper_plot_theme.jl"))
using .PaperPlotTheme

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
    xi, modes = freefree_modes_l2(5)
    n_modes = length(modes)
    beta_vals = vcat(0.0, 0.0, Surferbot.freefree_betaL_roots(n_modes - 2))

    y_all = vcat(modes...)
    ylim_abs = max(abs(minimum(y_all)), abs(maximum(y_all))) * 1.35
    ylims = (-ylim_abs, ylim_abs)

    outfile = joinpath(FIG_DIR, "plot_freefree_modes")
    PaperPlotTheme.with_theme() do
        fig = Figure(size = (700, 540), backgroundcolor = :white,
            figure_padding = (8, 20, 8, 8))
        for i in 1:n_modes
            n = i - 1
            is_bottom = i == n_modes
            ax = Axis(fig[i, 1]; limits = ((-0.5, 0.5), ylims),
                xticks = -0.5:0.25:0.5, yticks = [-2.0, 0.0, 2.0],
                xlabel = is_bottom ? L"x" : "", ylabel = i == 3 ? L"W_n(x)" : "",
                xlabelsize = 17, ylabelsize = 17, xticklabelsize = 15, yticklabelsize = 15,
                xticklabelsvisible = is_bottom, xticksvisible = is_bottom,
                xgridvisible = false, ygridvisible = false)
            hlines!(ax, [0.0]; color = (:black, 0.25), linewidth = 0.8)
            lines!(ax, xi, modes[i]; color = :black, linewidth = 1.8)
            label = LaTeXString(@sprintf("n = %d,\\; \\beta_n = %.2f", n, beta_vals[i]))
            text!(ax, -0.48, ylim_abs * 0.70;
                text = label,
                fontsize = 15, align = (:left, :center))
        end
        rowgap!(fig.layout, 0)
        save(outfile * ".pdf", fig)
        save(outfile * ".png", fig; px_per_unit = 2)
    end
    println("Saved $outfile.{pdf,png} (via CairoMakie)")
end

function main()
    mkpath(FIG_DIR)
    plot_modes()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
