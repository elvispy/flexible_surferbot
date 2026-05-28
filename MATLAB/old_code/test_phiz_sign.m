addpath('../src');
N=50; Omega=0.8; Nu=0; Gamma=0; M=1; xd=3; zd=1;
theta=0; zeta=1;

waveDispersion = @(k) k.*tanh(k*zd).*(1+Gamma*k.^2)+4i*Nu*k.^2 - Omega^2;
kstar = fsolve(waveDispersion, Omega^2, optimset('display','off'));

xs = linspace(-xd,xd,N);
zs = linspace(-zd,0,N);
dx = xs(2)-xs(1); dz = zs(2)-zs(1);
[X,Z] = meshgrid(xs,zs);

i1 = find(abs(xs)<=0.5,1); i2 = find(abs(xs)<=0.5,1,'last');

[~,Dz]  = getNonCompactFDmatrix2D(N,N,dx,dz,1,2);
[Dxx,Dzz] = getNonCompactFDmatrix2D(N,N,dx,dz,2,2);

[baseMatrix,boundaryMatrix] = build_matrices(N,i1,i2,Omega,kstar,dz,dx,Nu,Gamma);
boundaryIndices = find(boundaryMatrix);
[updatedMatrix,rhsVector] = update_matrices(Dxx+Dzz, xs, zs', N, boundaryIndices, baseMatrix, theta, zeta, dz);
phi_vec = updatedMatrix \ rhsVector;
Phi = reshape(phi_vec,N,N);
phiz_vec = Dz*phi_vec;
Phiz = reshape(phiz_vec,N,N);
phiz0 = Phiz(end,:);

W_raft = zeta + xs(i1:i2)*theta;
ratio = full(phiz0(i1:i2)) ./ (1i * W_raft);
fprintf('TEST 1: phiz0[raft]/(i*W_raft) mean=%.6f+%.6fi (should be 1+0i if hypothesis correct)\n', mean(real(ratio)), mean(imag(ratio)));

Dxx1 = getNonCompactFDmatrix(N,dx,2,2);
etaRaw_current = (Nu/Omega^2)*Dxx1 - 1i*speye(N);
etaRaw_correct = 1i*speye(N) - (Nu/Omega^2)*Dxx1;
for L_mat = {etaRaw_current, etaRaw_correct}
  L = L_mat{1};
  F = phiz0(:);
  L(1,:)=0; L(1,1)=-3/(2*dx)-1i*kstar; L(1,2)=2/dx; L(1,3)=-1/(2*dx); F(1)=0;
  L(end,:)=0; L(end,end)=3/(2*dx)+1i*kstar; L(end,end-1)=-2/dx; L(end,end-2)=1/(2*dx); F(end)=0;
  eta_sol = L\F;
  eta_raft = eta_sol(i1:i2);
  err = mean(abs(eta_raft - W_raft.'))/abs(zeta);
  err_neg = mean(abs(eta_raft + W_raft.'))/abs(zeta);
  fprintf('  Form [%s]: err_vs_+W=%.4f, err_vs_-W=%.4f\n', mat2str(L(N/2,N/2),4), err, err_neg);
end
