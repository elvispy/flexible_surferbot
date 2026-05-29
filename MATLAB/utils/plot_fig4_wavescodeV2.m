addpath('/Users/harrislab/Documents/GitHub/waves_code/MATLAB/old_code/waves_code')

L = 0.05; width = 0.03; g = 9.81; rho = 1000;
freq = 80; omega = 2*pi*freq; nu = 1e-6; gamma_st = 0.0722;
rho_raft = 0.052; mass = rho_raft * L;
motor_inertia = 0.13e-3 * 2.5e-3 * 2.0;
motor_pos = -0.003;

Omega = omega * sqrt(L/g);
Nu    = Omega * nu / sqrt(g * L^3);
Gamm  = gamma_st / (rho * g * L^2);
M_hat = mass / (rho * L^2 * width);
A_motor = motor_inertia / mass;
Fz0  = M_hat * Omega^2 * (A_motor / L);
xF0  = motor_pos / L;

N  = 300;
xd = 0.14 / (2*L);
zd = 0.00181 / L;

fprintf('Omega=%.4f  Nu=%.2e  Gamm=%.6f  M=%.5f  Fz0=%.5f  xF0=%.4f\n', ...
    Omega, Nu, Gamm, M_hat, Fz0, xF0);

% Step 1: solve symmetric case (xF0=0) for good initial guess
f0 = @(y) solver(y, Fz0, 0.0, Omega, Nu, Gamm, N, M_hat, xd, zd);
y0 = [0; 1e-4];
Y0 = fsolve(f0, y0, optimset('display','off','TolFun',1e-10,'TolX',1e-10));
fprintf('Symmetric: zeta=%.4f+%.4fi um\n', real(Y0(2))*L*1e6, imag(Y0(2))*L*1e6);

% Step 2: solve asymmetric case
f1 = @(y) solver(y, Fz0, xF0, Omega, Nu, Gamm, N, M_hat, xd, zd);
y1 = [1e-4; Y0(2)];
Y1 = fsolve(f1, y1, optimset('display','iter','TolFun',1e-10,'TolX',1e-10,'MaxFunEvals',500));
res = f1(Y1);
fprintf('Asymmetric: theta=%.4e+%.4ei  zeta=%.4f+%.4fi um  |res|=[%.2e,%.2e]\n', ...
    real(Y1(1)), imag(Y1(1)), real(Y1(2))*L*1e6, imag(Y1(2))*L*1e6, abs(res(1)), abs(res(2)));

[~,~,eta,~,X,~] = solver(Y1, Fz0, xF0, Omega, Nu, Gamm, N, M_hat, xd, zd);
xs = X(end,:);

% waves_code uses upward pressure convention -> eta_waves = -eta_Julia
% Flip sign to match Julia's convention
eta_plot = -full(eta);

x_cm      = xs * L * 100;
eta_um    = real(eta_plot) * L * 1e6;
x_contact = abs(xs) <= 0.5;

writematrix([xs(:), real(full(eta(:))), imag(full(eta(:)))], '/tmp/fig4_wavescodeV2_eta.csv');

fig = figure('Color','w','Position',[100 100 1100 420]);
plot(x_cm, eta_um, 'r', 'LineWidth', 1.2); hold on;
plot(x_cm(x_contact), eta_um(x_contact), 'b', 'LineWidth', 2.0);
xlabel('$x$ (cm)', 'Interpreter','latex','FontSize',20);
ylabel('$h$ ($\mu$m)', 'Interpreter','latex','FontSize',20);
xlim([-7 7]); ylim([-300 300]); yticks(-300:100:300);
grid on; box on;
set(gca,'FontSize',15,'TickLabelInterpreter','latex','FontName','Times New Roman');

print(fig, '/Users/harrislab/Documents/GitHub/waves_code/MATLAB/utils/figures/fig4_wavescodeV2', '-dpdf','-r0');
fprintf('Saved fig4_wavescodeV2.pdf\n');
