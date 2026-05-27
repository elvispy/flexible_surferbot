%% solver.m - Solves the Wave-Driven Propulsion Problem
% This function solves the wave-driven propulsion problem using finite difference methods.
% It computes the velocity potential, wave elevation, thrust, and other related quantities.
%
% INPUTS:
%   motionParams     - Vector containing raftAngle (rotation angle) and raftDisplacement (vertical displacement)
%   targetForceZ     - Target vertical force for Newton solver
%   targetForceX     - Target horizontal force position for Newton solver
%   oscillationFreq  - Dimensionless oscillation frequency (Omega)
%   viscosityCoeff   - Dimensionless viscosity (Nu)
%   surfaceTension   - Dimensionless surface tension (Gamm)
%   gridSize         - Number of grid points in each direction (N)
%   raftMass         - Dimensionless mass of the raft (M)
%   domainWidth      - Domain half-width in x-direction (xd)
%   domainDepth      - Domain depth in z-direction (zd)
%
% OUTPUTS:
%   solverResidual   - Residual vector for Newton solver
%   thrustForce      - Computed propulsion thrust force
%   waveElevation    - Computed surface wave elevation (eta)
%   velocityPotential - Velocity potential field (Phi)
%   xGridPoints      - Spatial coordinate matrix in x-direction (X)
%   zGridPoints      - Spatial coordinate matrix in z-direction (Z)

function [solverResidual, thrustForce, waveElevation, Phi, xGridPoints, zGridPoints] = solver(motionParams, targetForceZ, targetPositionX, oscillationFreq, viscosityCoeff, surfaceTension, gridSize, raftMass, domainWidth, domainDepth)

% Extract motion parameters
raftAngle = motionParams(1);  % Rotation angle of raft
raftDisplacement = motionParams(2);  % Vertical displacement of raft

% Calculate wavenumber from dispersion relation (Eqn 2.19 in the paper)
waveDispersion = @(waveNum) waveNum.*tanh(waveNum*domainDepth).*(1 + surfaceTension*waveNum.^2) + 4i*viscosityCoeff*waveNum.^2 - oscillationFreq^2;
wavenumber = fsolve(waveDispersion, oscillationFreq^2, optimset('display', 'off'));

% Create computational domain
xGridPoints = repmat(linspace(-domainWidth, domainWidth, gridSize), gridSize, 1);
zGridPoints = repmat(linspace(-domainDepth, 0, gridSize), gridSize, 1)';
surfaceElevationPoints = zGridPoints(:, end);
horizontalPositions = xGridPoints(end, :);

dx = horizontalPositions(2) - horizontalPositions(1);  % Grid spacing in x-direction
dz = surfaceElevationPoints(2) - surfaceElevationPoints(1);  % Grid spacing in z-direction

% Identify raft position in domain
raftLeftIdx = find(abs(horizontalPositions) <= 1/2, 1);
raftRightIdx = find(abs(horizontalPositions) <= 1/2, 1, 'last');

% Create finite difference matrices
[~, Dz] = getNonCompactFDmatrix2D(gridSize, gridSize, dx, dz, 1, 2);
[Dxx, Dzz] = getNonCompactFDmatrix2D(gridSize, gridSize, dx, dz, 2, 2);
Dx1 = getNonCompactFDmatrix(gridSize, dx, 1, 2);
Dxx1 = getNonCompactFDmatrix(gridSize, dx, 2, 2);

% Build operator matrix
[baseMatrix, boundaryMatrix] = build_matrices(gridSize, raftLeftIdx, raftRightIdx, oscillationFreq, wavenumber, dz, dx, viscosityCoeff, surfaceTension);
%[baseMatrix2, boundaryMatrix2] = build_matrices_readable(gridSize, raftLeftIdx, raftRightIdx, oscillationFreq, wavenumber, dz, dx, viscosityCoeff, surfaceTension);

% Apply boundary conditions
boundaryIndices = find(boundaryMatrix);
LaplacianMatrix = Dxx + Dzz;
[updatedMatrix, rhsVector] = update_matrices(LaplacianMatrix, horizontalPositions, surfaceElevationPoints, gridSize, boundaryIndices, baseMatrix, raftAngle, raftDisplacement, dz);

% Solve for velocity potential
phi = updatedMatrix \ rhsVector;
Phi = reshape(phi, gridSize, gridSize);
phiz = Dz * phi;
phizz = Dzz * phi;
Phiz = reshape(phiz, gridSize, gridSize);
Phizz = reshape(phizz, gridSize, gridSize);
phi0 = Phi(zGridPoints == 0);
phiz0 = Phiz(zGridPoints == 0);
phizz0 = Phizz(zGridPoints == 0);

% Construct eta (wave elevation) matrices
[etaMatrix, rhsEta] = eta_matrices(viscosityCoeff, oscillationFreq, Dxx1, gridSize, phiz0, wavenumber, dx, raftDisplacement, raftAngle, horizontalPositions, raftLeftIdx, raftRightIdx);

% Solve for wave elevation
waveElevation = etaMatrix \ rhsEta;
waveElevation(abs(horizontalPositions) <= 1/2) = (horizontalPositions(abs(horizontalPositions) <= 1/2) * raftAngle + raftDisplacement);

% Compute pressure on raft (eqn 2.20)
pressure = waveElevation + oscillationFreq^2 * 1i * phi0 + 2 * viscosityCoeff * phizz0;
pressure(abs(horizontalPositions') > 1/2) = 0;

% Compute lift and torque due to pressure
liftForce = trapz(horizontalPositions', pressure); % eqn 2.11b
torqueMoment = trapz(horizontalPositions', pressure .* horizontalPositions'); % appearing in eqn 2.11c

% Solving for F_{A, z} and x_A from Eqn 2.11b and 2.11c respectively
computedForceZ = (-raftMass * oscillationFreq^2 * raftDisplacement - liftForce);
x_A = 1 / computedForceZ * (-1/12 * raftMass * oscillationFreq^2 * raftAngle - torqueMoment);

% Compute propulsion thrust
thrustForce = 1/2 * trapz(horizontalPositions', (real(pressure) .* real(raftAngle) + imag(pressure) .* imag(raftAngle)));

% Output residuals for Newton solver
solverResidual(1) = computedForceZ - targetForceZ;
solverResidual(2) = x_A - targetPositionX;

end
