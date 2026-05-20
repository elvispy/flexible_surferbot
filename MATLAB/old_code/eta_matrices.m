%% eta_matrices.m - Constructs Finite Difference Matrices for Wave Elevation
% This function builds the linear system for solving the surface wave elevation, \eta,
% using finite difference methods. The matrix system is formulated based on the governing
% equations described in the paper.
%
% INPUTS:
%   viscosityCoeff   - Dimensionless viscosity (Nu)
%   oscillationFreq  - Dimensionless oscillation frequency (Omega)
%   secondDerivMatrix - Finite difference matrix for second derivatives in x (Dxx1)
%   gridSize        - Number of grid points in the domain (N)
%   verticalVelocity - Vertical velocity potential at the surface (phiz0)
%   waveNumber      - Computed wavenumber from the dispersion relation (kstar)
%   horizontalStep  - Grid spacing in the x-direction (dx)
%   raftDisplacement - Vertical displacement of the raft (zeta)
%   raftAngle       - Rotation angle of the raft (theta)
%   xGridPoints     - Horizontal grid points (xs)
%   raftLeftIdx     - Index marking the left boundary of the raft (i1)
%   raftRightIdx    - Index marking the right boundary of the raft (i2)
%
% OUTPUTS:
%   etaMatrix       - Sparse matrix representing the discretized wave elevation equation
%   rhsVector       - Right-hand side vector incorporating boundary conditions

function [etaMatrix, rhsVector] = eta_matrices(viscosityCoeff, oscillationFreq, secondDerivMatrix, gridSize, verticalVelocity, waveNumber, horizontalStep, raftDisplacement, raftAngle, xGridPoints, raftLeftIdx, raftRightIdx)

% Build operator matrix for wave elevation
etaMatrix = (viscosityCoeff / oscillationFreq^2) * secondDerivMatrix - 1i * speye(gridSize);

% Construct right-hand side vector
rhsVector = verticalVelocity;

% Apply left boundary condition (Equation 2.19b from paper)
leftBCIdx = find(etaMatrix(1, :));
etaMatrix(1, leftBCIdx) = 0;
etaMatrix(1, 1) = -3 / (2 * horizontalStep) - 1i * waveNumber;
etaMatrix(1, 2) = 2 / horizontalStep;
etaMatrix(1, 3) = -1 / (2 * horizontalStep);
rhsVector(1) = 0;

% Apply raft boundary condition at the left side (Equation 2.19c from paper)
raftLeftBCIdx = find(etaMatrix(raftLeftIdx, :));
etaMatrix(raftLeftIdx, raftLeftBCIdx) = 0;
etaMatrix(raftLeftIdx, raftLeftIdx) = 1;
rhsVector(raftLeftIdx) = raftDisplacement + xGridPoints(raftLeftIdx) * raftAngle;

% Apply right boundary condition (Equation 2.19d from paper)
rightBCIdx = find(etaMatrix(end, :));
etaMatrix(end, rightBCIdx) = 0;
etaMatrix(end, end) = 3 / (2 * horizontalStep) + 1i * waveNumber;
etaMatrix(end, end-1) = -2 / horizontalStep;
etaMatrix(end, end-2) = 1 / (2 * horizontalStep);
rhsVector(end) = 0;

% Apply raft boundary condition at the right side (Equation 2.19c from paper)
raftRightBCIdx = find(etaMatrix(raftRightIdx, :));
etaMatrix(raftRightIdx, raftRightBCIdx) = 0;
etaMatrix(raftRightIdx, raftRightIdx) = 1;
rhsVector(raftRightIdx) = raftDisplacement + xGridPoints(raftRightIdx) * raftAngle;

end
% 
% 
% function [L,F] = eta_matrices(Nu,Omega,Dxx1,N,phiz0,kstar,dx,zeta,theta,xs,i1,i2)
% 
% % Build operator
% L=(Nu/Omega^2)*Dxx1-speye(N);
% 
% % Construct right hand side
% F=phiz0;
% 
% % Apply left hand BC
% ids=find(L(1,:));
% L(1,ids)=0;
% L(1,1)=-3/(2*dx)-1i*kstar;
% L(1,2)=2/dx;
% L(1,3)=-1/(2*dx);
% F(1)=0;
% 
% % Apply raft BC (left)
% ids=find(L(i1,:));
% L(i1,ids)=0;
% L(i1,i1)=1;
% F(i1)=zeta+xs(i1)*theta;
% 
% % Apply right hand BC
% ids=find(L(end,:));
% L(end,ids)=0;
% L(end,end)=3/(2*dx)+1i*kstar;
% L(end,end-1)=-2/dx;
% L(end,end-2)=1/(2*dx);
% F(end)=0;
% 
% % Apply raft BC (right)
% ids=find(L(i2,:));
% L(i2,ids)=0;
% L(i2,i2)=1;
% F(i2)=zeta+xs(i2)*theta;
% 
% end