function [L,F] = eta_matrices(Nu,Omega,Dxx1,N,phiz0,kstar,dx,zeta,theta,xs,i1,i2)

% Build operator
L=1i*speye(N)-(Nu/Omega^2)*Dxx1;

% Construct right hand side
F=phiz0;

% Apply left hand BC
ids=find(L(1,:));
L(1,ids)=0;
L(1,1)=-3/(2*dx)-1i*kstar;
L(1,2)=2/dx;
L(1,3)=-1/(2*dx);
F(1)=0;

% Apply raft BC (left)
ids=find(L(i1,:));
L(i1,ids)=0;
L(i1,i1)=1;
F(i1)=zeta+xs(i1)*theta;

% Apply right hand BC
ids=find(L(end,:));
L(end,ids)=0;
L(end,end)=3/(2*dx)+1i*kstar;
L(end,end-1)=-2/dx;
L(end,end-2)=1/(2*dx);
F(end)=0;

% Apply raft BC (right)
ids=find(L(i2,:));
L(i2,ids)=0;
L(i2,i2)=1;
F(i2)=zeta+xs(i2)*theta;

end