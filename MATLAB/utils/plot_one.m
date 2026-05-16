addpath '../src'

names = {'U', 'x', 'z', 'phi', 'eta', 'args'};

% ── Exact parameters from Julia FlexibleParams (coupled sweep operating point) ──
% EI  = E*b*h^3/12  with E=3 GPa, b=3 cm, h=0.99 mm  (polycarbonate raft)
% I_m = m_rotor * r^2  with m=0.13 g, r=2.5 mm        (motor inertia)
% motor_position < 0  →  motor left of centre  →  rightward thrust (left→right)
[out{1:numel(names)}] = flexible_surferbot_v2( ...
    'sigma'         , 0.0722                        , ...  % [N/m]   surface tension
    'rho'           , 1000.0                        , ...  % [kg/m3] water density
    'omega'         , 2*pi * 80                     , ...  % [rad/s] 80 Hz
    'nu'            , 1e-6                          , ...  % [m2/s]  kinematic viscosity (water)
    'g'             , 9.81                          , ...  % [m/s2]  gravity
    'L_raft'        , 0.05                          , ...  % [m]     raft length
    'motor_position', -0.003                        , ...  % [m]     xA = -3 mm (Benham 2024 §3.2), left of centre → rightward
    'd'             , 0.03                          , ...  % [m]     raft depth (spanwise)
    'EI'            , Inf                            , ...  % [N m2]  rigid-raft limit
    'rho_raft'      , 0.052                         , ...  % [kg/m]  linear mass density
    'motor_inertia' , 0.13e-3 * 2.5e-3 * 2.0         , ...  % [kg m] m_rotor*r scaled ×2.0 to match Benham A=150µm
    'L_domain'      , 0.14                          , ...  % [m]     total domain = 14 cm → x ∈ [-7,7] cm
    'BC'            , 'radiative'                   );

S = cell2struct(out(:), names(:), 1);

% ── Publication figure ────────────────────────────────────────────────────────
x_c    = S.args.x_contact;
x_all  = S.x    * 1e2;          % m → cm
y_all  = real(S.eta) * 1e6;     % m → µm, t = 0 snapshot

fig = figure('Units', 'centimeters', 'Position', [2 2 17 6.5], ...
             'Color', 'w', 'PaperUnits', 'centimeters', ...
             'PaperSize', [17 6.5], 'PaperPosition', [0 0 17 6.5]);

ax = axes(fig);
hold(ax, 'on');

plot(ax, x_all,       y_all,       'r', 'LineWidth', 1.2);
plot(ax, x_all(x_c),  y_all(x_c),  'b', 'LineWidth', 2.0);

xlabel(ax, '$x$ (cm)',   'Interpreter', 'latex', 'FontSize', 20);
ylabel(ax, '$h$ ($\mu$m)', 'Interpreter', 'latex', 'FontSize', 20);
xlim(ax, [-7 7]);
ylim(ax, [-300 300]);
yticks(ax, -300:100:300);
set(ax, 'FontSize', 15, 'FontName', 'Times New Roman', ...
        'TickLabelInterpreter', 'latex', 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on');

outdir = '../utils/figures';
print(fig, fullfile(outdir, 'fig4_Aguero2026'), '-dpdf', '-r0');
fprintf('Saved fig4_Aguero2026.pdf\n');
