# Build paper figures from source scripts.
# Usage (run from this directory):
#   make         — build all figures
#   make sync    — copy paper figures to overleaf/img/ (reads main.tex for the list)
#   make clean   — remove generated figures
#
# Sweep data (JLD2/CSV) is NOT built here — submit the sweep scripts to the cluster first.
# If a required data file is missing, Make will error rather than silently skip.

JULIA    := julia --project=..
FIG_DIR  := ../output/figures
JLD2_DIR := ../output/jld2
CSV_DIR  := ../output/csv
OVERLEAF ?= $(HOME)/Documents/Github/overleaf/wave-driven-propulsion
TEX      := $(OVERLEAF)/main.tex

# ── Cluster-generated data (no build rules — must exist before make runs) ───
THRUST_CACHE    := $(JLD2_DIR)/thrust_sweeps.jld2
ALPHA_CACHE     := $(JLD2_DIR)/alpha_sweep_kappa_farfield.jld2
SWEEP_CPL_JLD2  := $(JLD2_DIR)/sweep_motor_position_EI_coupled_from_matlab.jld2
SWEEP_UCPL_JLD2 := $(JLD2_DIR)/sweep_motor_position_EI_uncoupled_from_matlab.jld2
SWEEP_CPL_CSV   := $(CSV_DIR)/sweeper_coupled_full_grid.csv
SWEEP_UCPL_CSV  := $(CSV_DIR)/sweeper_uncoupled_full_grid.csv

# ── One primary output per script (running the script produces all siblings) ─
FIGURES := \
	$(FIG_DIR)/plot_thrust_sweeps_xM.pdf \
	$(FIG_DIR)/plot_kappa_snapshot_1.95e-03.pdf \
	$(FIG_DIR)/plot_kappa_snapshot_grid_flexibility.pdf \
	$(FIG_DIR)/plot_kappa_snapshot_grid_motor_position.pdf \
	$(FIG_DIR)/plot_thrust_beam_and_LH_LH_cbrt.pdf \
	$(FIG_DIR)/plot_dimensionless_diagnostics_LH_cpl_theo.pdf \
	$(FIG_DIR)/plot_freefree_modes.pdf \
	$(FIG_DIR)/plot_beam_num_modes_cmp_cpl.pdf \
	$(FIG_DIR)/plot_fig10_modal_maps_3x3.pdf \
	$(FIG_DIR)/plot_fig4_Aguero2026_1.pdf

MSG ?= regenerate figures

.PHONY: all sync commit clean
all: $(FIGURES)

# ── Figure rules ─────────────────────────────────────────────────────────────
$(FIG_DIR)/plot_thrust_sweeps_xM.pdf: plot_thrust_sweeps.jl \
		$(THRUST_CACHE) $(ALPHA_CACHE) $(SWEEP_CPL_CSV)
	$(JULIA) --threads auto $<

$(FIG_DIR)/plot_kappa_snapshot_1.95e-03.pdf: plot_kappa_snapshot.jl
	$(JULIA) $< --paper-snapshots

$(FIG_DIR)/plot_kappa_snapshot_grid_flexibility.pdf: plot_kappa_snapshot.jl \
		$(THRUST_CACHE) $(ALPHA_CACHE) $(SWEEP_CPL_CSV)
	$(JULIA) $< --snapshot-grids

$(FIG_DIR)/plot_kappa_snapshot_grid_motor_position.pdf: $(FIG_DIR)/plot_kappa_snapshot_grid_flexibility.pdf

$(FIG_DIR)/plot_thrust_beam_and_LH_LH_cbrt.pdf: plot_thrust_beam_and_LH.jl \
		$(SWEEP_CPL_CSV)
	$(JULIA) --threads auto $<

$(FIG_DIR)/plot_dimensionless_diagnostics_LH_cpl_theo.pdf: plot_dimensionless_diagnostics_LH.jl \
		$(SWEEP_CPL_JLD2) $(SWEEP_UCPL_JLD2) $(SWEEP_CPL_CSV) $(SWEEP_UCPL_CSV)
	$(JULIA) --threads auto $<

$(FIG_DIR)/plot_freefree_modes.pdf: plot_freefree_modes.jl
	$(JULIA) $<

$(FIG_DIR)/plot_beam_num_modes_cmp_cpl.pdf: plot_beam_num_modes_cmp.jl \
		$(SWEEP_CPL_JLD2) $(SWEEP_UCPL_JLD2) $(SWEEP_CPL_CSV)
	$(JULIA) --threads auto $<

$(FIG_DIR)/plot_fig10_modal_maps_3x3.pdf: plot_fig10_modal_maps_3x3.jl \
		$(SWEEP_CPL_JLD2) $(SWEEP_UCPL_JLD2) $(SWEEP_CPL_CSV) $(SWEEP_UCPL_CSV)
	$(JULIA) --threads auto $<

$(FIG_DIR)/plot_fig4_Aguero2026_1.pdf: plot_fig4_Aguero2026.jl
	$(JULIA) $<

# ── Sync: copy only figures referenced in main.tex ───────────────────────────
sync:
	@perl -ne 'print "$$1\n" if /\{img\/([^}]+)\}/' $(TEX) | while read f; do \
		[ -f $(FIG_DIR)/$$f ] && cp $(FIG_DIR)/$$f $(OVERLEAF)/img/$$f && echo "synced $$f" || true; \
	done

# ── Commit: sync then record figures in both repos ───────────────────────────
# Usage: make commit MSG="retire Surferbot hlines"
commit: sync
	git -C $(OVERLEAF) add img/ && git -C $(OVERLEAF) diff --cached --quiet || \
		git -C $(OVERLEAF) commit -m "$(MSG)"
	git -C ../.. add output/figures/ && git -C ../.. diff --cached --quiet || \
		git -C ../.. commit -m "$(MSG)"

clean:
	rm -f $(FIGURES)
