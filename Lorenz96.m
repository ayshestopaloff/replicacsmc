% Generate sample data from Lorenz-96 model.

% Set random number generator seed
clear;
data_seed = RandStream('mt19937ar','Seed', 1);
RandStream.setGlobalStream(data_seed);

d = 3;
T = 2;

% Model parameters
F = 4.8801; % forcing parameter
sigmasq = 1e-2; % transition noise
interval = 0.1;
P = diag(ones(1,d-1),-1);
P(1,d) = 1;
PT = P';
PP = P*P;
Lorenz = @(x) ((PT*x' - PP*x').*(P*x') - x' + F)';

% Initial distribution
Initial = struct();
Initial.Mean = zeros(1,d);
Initial.Cov = sigmasq * eye(d);
Initial.Precision = inv(Initial.Cov);
Initial.Sample = @(n) mvnrnd(Initial.Mean, Initial.Cov, n); 

% Transition kernel
Transition = struct();
Transition.Mean = @(t,x,dir) RungeKutta4(x,dir,interval,Lorenz);
Transition.G = sigmasq * interval * eye(d);
Transition.Ginv = inv(Transition.G);
Transition.Sample = @(x,n) mvnrnd(Transition.Mean(1,x,1), Transition.G, n);

d_y = d;

% Generate observation
zetasq = 1e-3;
zeta = sqrt(zetasq);
Observation = struct();
Latentx = zeros(T+1,d);
Observation.y = zeros(T+1,d_y);
Latentx(1,:) = Initial.Sample(1);
Observation.Std = zeta;
Observation.Precision = 1 / (zetasq);
Observation.y(1,:) = Latentx(1,1:d_y) + Observation.Std * randn(1,d_y);

for t = 1:T
	Latentx(t+1,:) = Transition.Sample(Latentx(t,:), 1);
	Observation.y(t+1,:) = Latentx(t+1,1:d_y) + Observation.Std * randn(1,d_y);
end

C = Transition.G;
C_init = Initial.Cov;

save(strcat('lorenz_',num2str(d),'_', num2str(T), '.mat'));

