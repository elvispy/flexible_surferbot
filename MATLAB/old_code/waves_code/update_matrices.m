function [L,F] = update_matrices(Lap,rs,zs,N,BNDi,L0,theta,zeta,dz)

% Create spares right-hand side
F=sparse(N^2,1);

% Create the right-hand sides for each BC
LBC=0*zs;
BBC=0*rs;
RBC=0*zs;
TBC=-1i*dz*(zeta+rs*theta);
TBC(abs(rs)>1/2)=0;

% Top BC
is=N:N:N^2;
F(is)=TBC;
Lap(is,:)=0;

% Right BC
is=N^2-N+1:N^2;
F(is)=RBC;
Lap(is,:)=0;

% Bottom BC
is=1:N:N^2;
F(is)=BBC;
Lap(is,:)=0;

% Left BC
is=1:N;
F(is)=LBC;
Lap(is,:)=0;

% Update matrices
Lap(BNDi)=L0(BNDi);
L=Lap;

end