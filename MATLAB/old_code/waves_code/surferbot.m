close all
clear all

% Dimensional parameters (SI units)
L=5e-2;             % length of raft
width=0.03;         % width of raft
g=9.81;             % gravity
rho=1000;           % density of water
freq=80;            % oscillation frequency
omega=2*pi*freq;    % angular frequency
gamma=0.073;        % surface tension
nu=1e-6;            % kinematic viscosity
mass=2.6*1e-3;      % mass of surferbot
xF=-0.003;            % position of motor (measured from center)
Amp=-170.5*1e-6;       % amplitude of oscillations

% Dimensionless parameters
Omega=omega*sqrt(L/g);
Gamm=gamma/(rho*g*L^2);
Nu=Omega*nu/(g^(1/2)*L^(3/2));
M=mass/(rho*L^2*width);
xF0=xF/L;
Fz0=M*Omega^2*(Amp/L);

% Solver parameters
opts=optimset('display','iter');
N=300;                              % Number of grid points
y0=[1;1]*1e-4;                      % Initial guess for theta,zeta
xd=0.75;zd=0.25;                    % Size of domain

% Find theta and zeta using Newton
f=@(y) solver(y,Fz0,xF0,Omega,Nu,Gamm,N,M,xd,zd);
Y=fsolve(f,y0,opts);

% Extract thrust, wavefield and power
[~,thrust,eta,Phi,X,Z] = solver(Y,Fz0,xF0,Omega,Nu,Gamm,N,M,xd,zd);

% Calculate dimensional drift speed (drag-thrust balance)
Udim=(1/(1.33)*g*L^(3/2)*nu^(-1/2)*thrust)^(2/3)

% Create raft shape
xs=linspace(-xd,xd,N);
theta=Y(1);zeta=Y(2);
boat=real((zeta+xs*theta));
boat(abs(xs)>1/2)=NaN;

% Plot solution
fig1=figure(1);clf;
plt=pcolor(X,Z,imag(Phi));
set(plt,'edgecolor','none')
set(gca,'fontsize',20)
hold on
plot([-0.5,0.5],[0,0],'w','linewidth',4);
xlim([-xd,xd])
ylim([-zd,0])
xticks([-xd,0,xd])
yticks([-zd,0])
xlabel('$$x/L$$','interpreter','latex')
ylabel('$$z/L$$','interpreter','latex')
cl=colorbar('eastoutside');

fig2=figure(2);clf;hold on
plot(xs*L*1e2,real(eta)*L*1e6,'r-','linewidth',2)
plot(xs*L*1e2,real(boat)*L*1e6,'b','linewidth',4)
xlim([-xd,xd]*L*1e2)
ylim([-1,1]*250)
xlabel('$$x$$ (cm)','interpreter','latex')
ylabel('$$h$$ ($$\mu$$m)','interpreter','latex')
set(gca,'fontsize',20)

set(fig1,'position',[40 376 500 200])
set(fig2,'position',[640 376 500 200])
