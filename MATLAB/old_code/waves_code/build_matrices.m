function [L,BND] = build_matrices(N,i1,i2,Omega,kstar,dz,dx,Nu,Gamm)

% Build sparse matrices
L=sparse(N^2,N^2);
BND=sparse(N^2,N^2);

% Set left and right hand wavenumbers
kstarL=kstar;
kstarR=-kstar;

% TBC
j=0;
for i=N:N:N^2
    j=j+1;
    id=find(L(i,:));
    L(i,id)=0;
    
    if  and(i>=i1*N,i<=i2*N)
        L(i,i)=-3/2;
        L(i,i-1)=2;
        L(i,i-2)=-1/2;
        BND(i,i)=1;
        BND(i,i-1)=1;
        BND(i,i-2)=1;
    else
        if i-N<=0 % Left hand point
            L(i,i)=(-3/2).*dz.^(-1)+(-3).*dx.^(-2).*dz.^(-1).*Gamm+(sqrt(-1)*(-8)).*dx.^(-2).*Nu+Omega.^2;
            L(i,i-1)=2.*dz.^(-1)+4.*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i-2)=(-1/2).*dz.^(-1)+(-1).*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i+N)=(-15/2).*dx.^(-2).*dz.^(-1).*Gamm+(sqrt(-1)*(-20)).*dx.^(-2).*Nu;
            L(i,i+2*N)=6.*dx.^(-2).*dz.^(-1).*Gamm+(sqrt(-1)*16).*dx.^(-2).*Nu;
            L(i,i+3*N)=(-3/2).*dx.^(-2).*dz.^(-1).*Gamm+(sqrt(-1)*(-4)).*dx.^(-2).*Nu;
            L(i,i+N-1)=10.*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i+2*N-1)=(-8).*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i+3*N-1)=2.*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i+N-2)=(-5/2).*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i+2*N-2)=2.*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i+3*N-2)=(-1/2).*dx.^(-2).*dz.^(-1).*Gamm;
            BND(i,i)=1;
            BND(i,i-1)=1;
            BND(i,i-2)=1;
            BND(i,i+N)=1;
            BND(i,i+2*N)=1;
            BND(i,i+3*N)=1;
            BND(i,i+N-1)=1;
            BND(i,i+2*N-1)=1;
            BND(i,i+3*N-1)=1;
            BND(i,i+N-2)=1;
            BND(i,i+2*N-2)=1;
            BND(i,i+3*N-2)=1;
        elseif i+N>N^2 % Right hand point
            L(i,i)=(-3/2).*dz.^(-1)+(-3).*dx.^(-2).*dz.^(-1).*Gamm+(sqrt(-1)*(-8)).*dx.^(-2).*Nu+Omega.^2;
            L(i,i-1)=2.*dz.^(-1)+4.*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i-2)=(-1/2).*dz.^(-1)+(-1).*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i-N)=(-15/2).*dx.^(-2).*dz.^(-1).*Gamm+(sqrt(-1)*(-20)).*dx.^(-2).*Nu;
            L(i,i-2*N)=6.*dx.^(-2).*dz.^(-1).*Gamm+(sqrt(-1)*16).*dx.^(-2).*Nu;
            L(i,i-3*N)=(-3/2).*dx.^(-2).*dz.^(-1).*Gamm+(sqrt(-1)*(-4)).*dx.^(-2).*Nu;
            L(i,i-N-1)=10.*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i-2*N-1)=(-8).*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i-3*N-1)=2.*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i-N-2)=(-5/2).*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i-2*N-2)=2.*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i-3*N-2)=(-1/2).*dx.^(-2).*dz.^(-1).*Gamm;
            BND(i,i)=1;
            BND(i,i-1)=1;
            BND(i,i-2)=1;
            BND(i,i-N)=1;
            BND(i,i-2*N)=1;
            BND(i,i-3*N)=1;
            BND(i,i-N-1)=1;
            BND(i,i-2*N-1)=1;
            BND(i,i-3*N-1)=1;
            BND(i,i-N-2)=1;
            BND(i,i-2*N-2)=1;
            BND(i,i-3*N-2)=1;
        else
            L(i,i)=(-3/2).*dz.^(-1)+(-3).*dx.^(-2).*dz.^(-1).*Gamm+(sqrt(-1)*(-8)).*dx.^(-2).*Nu+Omega.^2;
            L(i,i-1)=2.*dz.^(-1)+4.*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i-2)=(-1/2).*dz.^(-1)+(-1).*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i+N)=(3/2).*dx.^(-2).*dz.^(-1).*Gamm+(sqrt(-1)*4).*dx.^(-2).*Nu;
            L(i,i-N)=(3/2).*dx.^(-2).*dz.^(-1).*Gamm+(sqrt(-1)*4).*dx.^(-2).*Nu;
            L(i,i+N-1)=(-2).*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i-N-1)=(-2).*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i+N-2)=(1/2).*dx.^(-2).*dz.^(-1).*Gamm;
            L(i,i-N-2)=(1/2).*dx.^(-2).*dz.^(-1).*Gamm;
            BND(i,i)=1;
            BND(i,i-1)=1;
            BND(i,i-2)=1;
            BND(i,i+N)=1;
            BND(i,i-N)=1;
            BND(i,i+N-1)=1;
            BND(i,i-N-1)=1;
            BND(i,i+N-2)=1;
            BND(i,i-N-2)=1;
        end
    end
    
end


% RBC
j=0;
for i=N^2-N+1:N^2
    j=j+1;
    id=find(L(i,:));
    L(i,id)=0;
    
    L(i,i)=-3/2-1i*kstarL*dx;
    L(i,i-N)=2;
    L(i,i-2*N)=-1/2;
    
    BND(i,i)=1;
    BND(i,i-N)=1;
    BND(i,i-2*N)=1;
end

% BBC
j=0;
for i=1:N:N^2
    j=j+1;
    id=find(L(i,:));
    L(i,id)=0;
    
    L(i,i)=3/2;
    L(i,i+1)=-2;
    L(i,i+2)=1/2;
    
    BND(i,i)=1;
    BND(i,i+1)=1;
    BND(i,i+2)=1;
end

% LBC
j=0;
for i=1:N
    j=j+1;
    id=find(L(i,:));
    L(i,id)=0;
    
    L(i,i)=3/2-1i*kstarR*dx;
    L(i,i+N)=-2;
    L(i,i+2*N)=1/2;
    
    BND(i,i)=1;
    BND(i,i+N)=1;
    BND(i,i+2*N)=1;
end

end