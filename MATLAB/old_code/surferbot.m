%% SurferBot Wave-Driven Propulsion Simulation
% This script models the wave-driven propulsion of SurferBot, a small raft
% oscillating at a fluid interface. The equations are based on the theoretical
% framework described in the paper "On wave-driven propulsion."

close all
clear all

%% Dimensional Parameters (SI Units)
raftLength = 1;      % Length of the raft (m)
raftWidth = 0.03;       % Width of the raft (m)
gravity = 9.81;         % Gravitational acceleration (m/s^2)
fluidDensity = 1000;    % Density of water (kg/m^3)
oscillationFreq = 1;   % Oscillation frequency (Hz)
angularFreq = 2 * pi * oscillationFreq; % Angular frequency (rad/s)
surfaceTension = 0.073; % Surface tension of water (N/m)
kinematicViscosity = 1e-6; % Kinematic viscosity (m^2/s)
raftMass = 2.6e-3;      % Mass of SurferBot (kg)
motorPosition = 0*-0.003; % Position of motor relative to raft center (m)
oscillationAmp = 152e-3;% Amplitude of oscillations (m)

raftLength = 5e-2;      % Length of the raft (m)
raftWidth = 0.03;       % Width of the raft (m)
gravity = 9.81;         % Gravitational acceleration (m/s^2)
fluidDensity = 1000;    % Density of water (kg/m^3)
oscillationFreq = 5;   % Oscillation frequency (Hz)
angularFreq = 2 * pi * oscillationFreq; % Angular frequency (rad/s)
surfaceTension = 0*0.073; % Surface tension of water (N/m)
kinematicViscosity = 0*1e-6; % Kinematic viscosity (m^2/s)
raftMass = 0.009;      % Mass of SurferBot (kg)
motorPosition = 0; % Position of motor relative to raft center (m)
oscillationAmp = 100*0.13e-3 * 2.5e-3;% Amplitude of oscillations (m)



%% Dimensionless Parameters (following paper's notation)
dimlessOscillationFreq = angularFreq * sqrt(raftLength / gravity); % Omega
dimlessSurfaceTension = surfaceTension / (fluidDensity * gravity * raftLength^2); % Gamm
dimlessViscosity = dimlessOscillationFreq * kinematicViscosity / (gravity^(1/2) * raftLength^(3/2)); % Nu
dimlessMass = raftMass / (fluidDensity * raftLength^2 * raftWidth); % M
dimlessMotorPos = motorPosition / raftLength; % xF0
dimlessForceZ = 0.01 / (raftMass * angularFreq^2); %
dimlessForceZ = dimlessMass * dimlessOscillationFreq^2 * (oscillationAmp / raftLength); % Fz0 
%0.0036935 -  0.0095208i; %
%% Solver Parameters
solverOptions = optimset('display', 'iter'); % Solver display options
gridSize = 300; % Number of grid points
initialGuess = [1; 1] * 1e-4; % Initial guess for raftAngle and raftDisplacement
domainWidth = 3; domainDepth = 1; % Domain size (scaled by raftLength)
domainWidth = 10; domainDepth = 10; % Domain size (scaled by raftLength)

%% Find Raft Motion Parameters using Newton Solver
solverFunction = @(motionParams) solver(motionParams, dimlessForceZ, dimlessMotorPos, dimlessOscillationFreq, dimlessViscosity, dimlessSurfaceTension, gridSize, dimlessMass, domainWidth, domainDepth);
motionSolution = fsolve(solverFunction, initialGuess, solverOptions);

%% Extract Thrust, Wave Field, and Power Output
[~, thrustForce, waveElevation, velocityPotential, xGridPoints, zGridPoints] = ...
    solver(motionSolution, dimlessForceZ, dimlessMotorPos, dimlessOscillationFreq, dimlessViscosity, dimlessSurfaceTension, gridSize, dimlessMass, domainWidth, domainDepth);

%% Calculate Dimensional Drift Speed (Using Thrust-Drag Balance)
dimensionalSpeed = (1 / 1.33 * gravity * raftLength^(3/2) * kinematicViscosity^(-1/2) * thrustForce)^(2/3);

%% Create Raft Shape for Visualization
horizontalPositions = linspace(-domainWidth, domainWidth, gridSize); % X-coordinates
raftAngle = motionSolution(1); raftDisplacement = motionSolution(2);
raftShape = real((raftDisplacement + horizontalPositions * raftAngle)); % Raft shape as function of angle and displacement
raftShape(abs(horizontalPositions) > 1/2) = NaN; % Remove out-of-domain points

%% Plot the Velocity Potential Field
fig1 = figure(1); clf;

colormap winter
plt = pcolor(xGridPoints, zGridPoints, imag(velocityPotential) / (angularFreq * raftLength^2));

shading interp
set(plt, 'edgecolor', 'none')
set(gca, 'fontsize', 20)
hold on
plot([-0.5, 0.5], [0, 0], 'w', 'linewidth', 4);
xlim([-domainWidth, domainWidth])
ylim([-domainDepth, 0])
xticks([-domainWidth, 0, domainWidth])
yticks([-domainDepth, 0])
xlabel('$$x/L$$', 'interpreter', 'latex')
ylabel('$$z/L$$', 'interpreter', 'latex')
cl = colorbar('eastoutside');

contour(xGridPoints, zGridPoints, imag(velocityPotential) / (angularFreq * raftLength^2), ...
        8, 'k');            % 10 contour levels, black lines
set(fig1, 'position', [40 376 800 200])
    
%% Plot the Wave Elevation and Raft Shape
fig2 = figure(2); clf; hold on
plot(horizontalPositions * raftLength * 1e2, real(waveElevation) * raftLength * 1e6, 'r-', 'linewidth', 2) % Wave elevation
plot(horizontalPositions * raftLength * 1e2, real(raftShape) * raftLength * 1e6, 'b', 'linewidth', 4) % Raft shape
xlim([-domainWidth, domainWidth] * raftLength * 1e2)
ylim([-1, 1] * 250)
xlabel('$$x$$ (cm)', 'interpreter', 'latex')
ylabel('$$h$$ ($$\mu$$m)', 'interpreter', 'latex')
set(gca, 'fontsize', 20)


%% Adjust Figure Positions for Better Viewing

set(fig2, 'position', [640 376 500 200])
%close(fig2)