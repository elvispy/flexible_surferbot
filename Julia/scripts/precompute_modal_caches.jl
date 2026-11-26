# Fill modal_pressure_maps.jld2 for the truncations used by the ROM figures (N = 2, 4, 8).
include(joinpath(@__DIR__, "plot_dimensionless_diagnostics_LH.jl"))
using Surferbot
for (name, preset) in (("coupled", Surferbot.default_coupled_motor_position_EI_sweep),)
    fp = coerce_flexible_params(preset().base_params)
    for N in (2, 4, 8)
        ModalPressureMap.load_or_compute_modal_pressure_map(fp; output_dir=joinpath(@__DIR__, "..", "output"), num_modes_basis=N)
        println("cached N=$N")
    end
end
