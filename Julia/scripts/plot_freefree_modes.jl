using Surferbot
using CairoMakie
using LaTeXStrings
using Printf

const FIG_DIR = joinpath(@__DIR__, "../output/figures")
const NEWCM_DIR = "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/newcomputermodern"
const NEWCM_FONT = joinpath(NEWCM_DIR, "NewCM10-Regular.otf")
const NEWCM_MATH = joinpath(NEWCM_DIR, "NewCMMath-Regular.otf")

function setup_newcm_mathfonts()
    MTE_ID = Base.PkgId(Base.UUID("0a4f8689-d25c-4efe-a92b-7142dfc1aa53"), "MathTeXEngine")
    MTE = get(Base.loaded_modules, MTE_ID, nothing)
    MTE === nothing && return
    isfile(NEWCM_MATH) || return
    try
        MTE.set_texfont_family!(
            regular    = joinpath(NEWCM_DIR, "NewCM10-Regular.otf"),
            italic     = joinpath(NEWCM_DIR, "NewCM10-Italic.otf"),
            bold       = joinpath(NEWCM_DIR, "NewCM10-Bold.otf"),
            bolditalic = joinpath(NEWCM_DIR, "NewCM10-BoldItalic.otf"),
            math       = NEWCM_MATH,
        )
    catch
    end
end

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
    setup_newcm_mathfonts()
    
    xi, modes = freefree_modes_l2(5)
    n_modes = length(modes)
    beta_vals = vcat(0.0, 0.0, Surferbot.freefree_betaL_roots(n_modes - 2))

    y_all = vcat(modes...)
    ylim_abs = max(abs(minimum(y_all)), abs(maximum(y_all))) * 1.35
    ylims = (-ylim_abs, ylim_abs)

    set_theme!(Theme(
        fonts = (; regular = NEWCM_FONT),
        fontsize = 18,
        Axis = (;
            xlabelsize = 20, ylabelsize = 20,
            xticklabelsize = 18, yticklabelsize = 18,
            xgridvisible = false, ygridvisible = false,
        ),
    ))

    fig = Figure(size = (700, 540), backgroundcolor = :white, figure_padding = (10, 35, 10, 10))

    axes = Vector{Axis}(undef, n_modes)
    for i in 1:n_modes
        n = i - 1
        is_top    = (i == 1)
        is_bottom = (i == n_modes)
        ax = Axis(fig[i, 1];
            limits = ((-0.5, 0.5), ylims),
            xticks = -0.5:0.25:0.5,
            yticks = ([-1.0, 0.0, 1.0], [L"-1", L"0", L"1"]),
            xlabel = is_bottom ? L"\xi = x/L" : "",
            xticklabelsvisible = is_bottom,
            xticksvisible = is_bottom,
            topspinevisible  = is_top,
            bottomspinevisible = true,
            leftspinevisible = true,
            rightspinevisible = true,
        )
        axes[i] = ax
        hlines!(ax, [0.0]; color = (:black, 0.25), linewidth = 0.8)
        lines!(ax, xi, modes[i]; color = :black, linewidth = 1.8)
        label_y = ylim_abs * 0.90
        label = @sprintf("n = %d,\\; \\beta_n = %.2f", n, beta_vals[i])
        text!(ax, -0.48, label_y;
            text = latexstring(label),
            fontsize = 20, font = NEWCM_FONT,
            align = (:left, :top))
    end

    # Tight stacking: equal rows, zero gap
    for i in 1:n_modes
        rowsize!(fig.layout, i, Fixed(88))
    end
    rowgap!(fig.layout, 0)

    Label(fig[:, 0], L"W_n(\xi)"; rotation = pi/2, tellheight = false, fontsize = 20)

    return fig
end

function main()
    mkpath(FIG_DIR)
    fig = plot_modes()
    outbase = joinpath(FIG_DIR, "plot_freefree_modes")
    save(outbase * ".pdf", fig)
    save(outbase * ".png", fig; px_per_unit = 2)
    println("Saved $outbase.{pdf,png}")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
