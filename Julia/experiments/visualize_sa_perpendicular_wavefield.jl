#!/usr/bin/env julia

using CairoMakie
using LaTeXStrings
using LinearAlgebra

const S_AMP = 1.0 + 0.0im
const A_AMP = 0.0 + 1.0im
const RAFT_HALF_LENGTH = 0.5
const DOMAIN_HALF_LENGTH = 4.5
const WAVENUMBER = 2pi / 1.15
const NFRAMES = 180
const FRAMERATE = 30

function complex_fields(x::AbstractVector{<:Real})
    symmetric = similar(x, ComplexF64)
    antisymmetric = similar(x, ComplexF64)

    for (j, xj) in pairs(x)
        if xj < -RAFT_HALF_LENGTH
            distance = -xj - RAFT_HALF_LENGTH
            phase = cis(-WAVENUMBER * distance)
            symmetric[j] = S_AMP * phase
            antisymmetric[j] = -A_AMP * phase
        elseif xj > RAFT_HALF_LENGTH
            distance = xj - RAFT_HALF_LENGTH
            phase = cis(-WAVENUMBER * distance)
            symmetric[j] = S_AMP * phase
            antisymmetric[j] = A_AMP * phase
        else
            symmetric[j] = S_AMP
            antisymmetric[j] = 2xj * A_AMP
        end
    end

    return symmetric, antisymmetric, symmetric + antisymmetric
end

function phasor_segment(z::Complex)
    Point2f[(0, 0), (real(z), imag(z))]
end

function main()
    eta_left = S_AMP - A_AMP
    eta_right = S_AMP + A_AMP
    @assert isapprox(real(S_AMP * conj(A_AMP)), 0.0; atol=1e-14)
    @assert isapprox(abs(S_AMP), 1.0; atol=1e-14)
    @assert isapprox(abs(A_AMP), 1.0; atol=1e-14)
    @assert isapprox(abs(eta_left), abs(eta_right); rtol=1e-14)
    @assert isapprox(abs(eta_left), sqrt(2); rtol=1e-14)

    x = collect(range(-DOMAIN_HALF_LENGTH, DOMAIN_HALF_LENGTH; length=1601))
    symmetric_hat, antisymmetric_hat, total_hat = complex_fields(x)
    theta = Observable(0.0)

    symmetric = @lift real.(symmetric_hat .* cis($theta))
    antisymmetric = @lift real.(antisymmetric_hat .* cis($theta))
    total = @lift real.(total_hat .* cis($theta))

    left_now = @lift eta_left * cis($theta)
    right_now = @lift eta_right * cis($theta)
    s_now = @lift S_AMP * cis($theta)
    plus_a_now = @lift A_AMP * cis($theta)
    minus_a_now = @lift -A_AMP * cis($theta)

    fig = Figure(size=(1280, 820), fontsize=19, backgroundcolor=:white)
    Label(fig[0, 1:2], L"S \perp A,\quad |S|=|A|=1"; fontsize=27)

    ax_wave = Axis(
        fig[1, 1:2],
        xlabel=L"x/L",
        ylabel=L"\eta(x,t)",
        limits=((-DOMAIN_HALF_LENGTH, DOMAIN_HALF_LENGTH), (-2.1, 2.1)),
        xticks=-4:1:4,
        yticks=-2:1:2,
    )
    vspan!(ax_wave, -RAFT_HALF_LENGTH, RAFT_HALF_LENGTH; color=(:gray70, 0.35))
    hlines!(ax_wave, [0.0]; color=:gray65, linewidth=1)
    lines!(ax_wave, x, symmetric; color=:dodgerblue3, linewidth=2, linestyle=:dash, label=L"S")
    lines!(ax_wave, x, antisymmetric; color=:red3, linewidth=2, linestyle=:dot, label=L"A")
    lines!(ax_wave, x, total; color=:black, linewidth=3, label=L"S\pm A")
    axislegend(ax_wave; position=:rt, framevisible=true)
    text!(ax_wave, 0, -1.85; text="raft", align=(:center, :center), color=:gray35, fontsize=16)
    text!(ax_wave, -2.6, 1.78; text=L"S-A", align=(:center, :center), fontsize=19)
    text!(ax_wave, 2.6, 1.78; text=L"S+A", align=(:center, :center), fontsize=19)

    ax_phase = Axis(
        fig[2, 1],
        xlabel="real part",
        ylabel="imaginary part",
        title="Complex amplitudes at the raft edges",
        aspect=DataAspect(),
        limits=((-1.8, 1.8), (-1.8, 1.8)),
        xticks=-1:1:1,
        yticks=-1:1:1,
    )
    hlines!(ax_phase, [0.0]; color=:gray75, linewidth=1)
    vlines!(ax_phase, [0.0]; color=:gray75, linewidth=1)
    lines!(ax_phase, @lift(phasor_segment($s_now)); color=:dodgerblue3, linewidth=3, label=L"S")
    lines!(ax_phase, @lift(phasor_segment($plus_a_now)); color=:red3, linewidth=3, label=L"+A")
    lines!(ax_phase, @lift(phasor_segment($minus_a_now)); color=:red3, linewidth=2, linestyle=:dash, label=L"-A")
    lines!(ax_phase, @lift(phasor_segment($right_now)); color=:black, linewidth=4, label=L"S+A")
    lines!(ax_phase, @lift(phasor_segment($left_now)); color=:gray35, linewidth=4, label=L"S-A")
    scatter!(ax_phase, @lift([Point2f(real($right_now), imag($right_now))]); color=:black, markersize=12)
    scatter!(ax_phase, @lift([Point2f(real($left_now), imag($left_now))]); color=:gray35, markersize=12)
    axislegend(ax_phase; position=:rb, nbanks=2, framevisible=true, labelsize=14)

    ax_amp = Axis(
        fig[2, 2],
        ylabel="outgoing amplitude",
        title="The two sides carry equal wave intensity",
        limits=((0.4, 2.6), (0.0, 1.65)),
        xticks=([1.0, 2.0], [L"|S-A|", L"|S+A|"]),
        yticks=0:0.5:1.5,
    )
    barplot!(ax_amp, [1.0, 2.0], [abs(eta_left), abs(eta_right)];
        color=[:gray35, :black], width=0.55)
    hlines!(ax_amp, [sqrt(2)]; color=:gray55, linestyle=:dash, linewidth=1.5)
    text!(ax_amp, 1.5, 1.50; text=L"\sqrt{2}", align=(:center, :bottom), fontsize=19)
    text!(ax_amp, 1.5, 0.23;
        text=L"|S-A|^2=|S+A|^2=|S|^2+|A|^2",
        align=(:center, :center), fontsize=18)

    rowgap!(fig.layout, 18)
    colgap!(fig.layout, 30)

    output = normpath(joinpath(@__DIR__, "..", "output", "figures",
        "visualize_sa_perpendicular_wavefield.mp4"))
    mkpath(dirname(output))
    record(fig, output, 1:NFRAMES; framerate=FRAMERATE) do frame
        theta[] = 2pi * (frame - 1) / NFRAMES
    end

    println("Wrote $output")
    println("|S-A| = $(abs(eta_left)), |S+A| = $(abs(eta_right))")
end

main()
