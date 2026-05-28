function [out,thrust,eta,Phi,X,Z] = solver(y,Fz0,xF0,Omega,Nu,Gamm,N,M,xd,zd)
theta=y(1);zeta=y(2);

% Calculate wavenumber from dispersion relation
fk=@(k) k.*tanh(k*zd).*(1+Gamm*k.^2)+4i*Nu*k.^2-Omega^2;
kstar=fsolve(fk,Omega^2,optimset('display','off'));

% Create domain
X=repmat(linspace(-xd,xd,N),N,1);
Z=repmat(linspace(-zd,0,N),N,1)';
zs=Z(:,end);
xs=X(end,:);
dx=xs(2)-xs(1);
dz=zs(2)-zs(1);

% Find raft position
i1=find(abs(xs)<=1/2,1);
i2=find(abs(xs)<=1/2,1,'last');

% Create finite difference matrices
[~, Dz] = getNonCompactFDmatrix2D(N,N,dx,dz,1,2);
[Dxx, Dzz] = getNonCompactFDmatrix2D(N,N,dx,dz,2,2);
Dx1 = getNonCompactFDmatrix(N,dx,1,2);
Dxx1 = getNonCompactFDmatrix(N,dx,2,2);

% Build operator matrix
[L0,BND]=build_matrices(N,i1,i2,Omega,kstar,dz,dx,Nu,Gamm);

% Apply boundary conditions
BNDi=find(BND);
Lap=Dxx+Dzz;
[L,F] = update_matrices(Lap,xs,zs,N,BNDi,L0,theta,zeta,dz);

% Invert system
phi=L\F;

% Extract solution and derivatives of solution
Phi=reshape(phi,N,N);
phiz=Dz*phi;
phizz=Dzz*phi;
Phiz=reshape(phiz,N,N);
Phizz=reshape(phizz,N,N);
phi0=(Phi(Z==0));
phiz0=(Phiz(Z==0));
phizz0=(Phizz(Z==0));

% Build eta matrices
[Leta,Feta]=eta_matrices(Nu,Omega,Dxx1,N,phiz0,kstar,dx,zeta,theta,xs,i1,i2);

% Invert eta system
eta=Leta\Feta;

% Set (wavefield = raft position) on the raft
eta(abs(xs)<=1/2)=(xs(abs(xs)<=1/2)*theta+zeta);

% Calculate the pressure on the raft
pres=-(eta+Omega^2*1i*phi0+2*Nu*phizz0);
pres(abs(xs')>1/2)=0;

% Calculate the lift and torque from the pressure
lift=trapz(xs',pres);
torque=trapz(xs',pres.*(xs'));

% use first-order dynamics to calculate Fz and xF
Fz=(-M*Omega^2*zeta-lift);
xF=1/Fz*(-1/12*M*Omega^2*theta-torque);

% Construct thrust
thrust=-1/2*trapz(xs',(real(pres).*real(theta)+imag(pres).*imag(theta)));

% Set output variables for Newton solver
out(1)=Fz-Fz0;
out(2)=xF-xF0;
end