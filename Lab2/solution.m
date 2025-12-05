clear; clc; close all;
rng(0);  % for reproducibility

%  TASK 1: 1D APPROXIMATION

fprintf(' TASK 1: 1D function approximation \n');

% Generating training data (20 samples in [0, 1])
N = 20;
x = linspace(0, 1, N)';   % column vector (20x1)

% Given target function:
% y = (1 + 0.6 * sin(2*pi*x/0.7) + 0.3 * sin(2*pi*x)) / 2;
t = (1 + 0.6 * sin(2*pi*x/0.7) + 0.3 * sin(2*pi*x)) / 2;  % (20x1)

% Network architecture for 1D
n_in_1D  = 1;
n_hid_1D = 6;    % between 4 and 8
n_out_1D = 1;

% Initializing weights and biases (small random values)
Wh_1D = 0.5 * randn(n_hid_1D, n_in_1D);   % (6x1)
bh_1D = 0.5 * randn(n_hid_1D, 1);         % (6x1)

Wo_1D = 0.5 * randn(n_out_1D, n_hid_1D);  % (1x6)
bo_1D = 0.5 * randn(n_out_1D, 1);         % (1x1)

% Training parameters
eta_1D      = 0.05;     % learning rate
n_epochs_1D = 2000;     % number of epochs

E_hist_1D = zeros(n_epochs_1D, 1);  % MSE per epoch

% Training loop (backpropagation, online learning)
for epoch = 1:n_epochs_1D
    sq_err_sum = 0;
    
    for i = 1:N
        xi = x(i);      % scalar input
        ti = t(i);      % scalar target
        
        zh = Wh_1D * xi + bh_1D;    % (6x1)
        ah = tanh(zh);              % tanh activation
        
        zo = Wo_1D * ah + bo_1D;    % (1x1)
        yi = zo;                    % linear
        
        err = yi - ti;              % scalar
        sq_err_sum = sq_err_sum + 0.5*err^2;
        
        % Output delta
        delta_out = err;            % linear output
        
        % Hidden deltas
        delta_hid = (Wo_1D' * delta_out) .* (1 - ah.^2);  % (6x1)
        
        % Gradients 
        dWo = delta_out * ah';  % (1x6)
        dbo = delta_out;        % scalar
        
        dWh = delta_hid * xi;   % (6x1)
        dbh = delta_hid;        % (6x1)
        
        %  Update 
        Wo_1D = Wo_1D - eta_1D * dWo;
        bo_1D = bo_1D - eta_1D * dbo;
        
        Wh_1D = Wh_1D - eta_1D * dWh;
        bh_1D = bh_1D - eta_1D * dbh;
    end
    
    % Mean squared error for this epoch
    E_hist_1D(epoch) = sq_err_sum / N;
    
    if mod(epoch, 200) == 0
        fprintf('1D Epoch %4d, MSE = %.6f\n', epoch, E_hist_1D(epoch));
    end
end

% Evaluating trained 1D network on dense grid
x_dense = linspace(0, 1, 200)';   % more points for smooth plot
t_dense = (1 + 0.6 * sin(2*pi*x_dense/0.7) + 0.3 * sin(2*pi*x_dense)) / 2;

y_dense = zeros(size(x_dense));

for i = 1:length(x_dense)
    xi = x_dense(i);
    zh = Wh_1D * xi + bh_1D;
    ah = tanh(zh);
    zo = Wo_1D * ah + bo_1D;
    y_dense(i) = zo;
end

% Plots for 1D task
figure;
plot(E_hist_1D, 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Mean squared error');
title('Task 1: 1D MLP – Training error (MSE)');
grid on;

figure;
plot(x_dense, t_dense, 'LineWidth', 1.5); hold on;
plot(x_dense, y_dense, '--', 'LineWidth', 1.5);
plot(x, t, 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'auto');
legend('Target function', 'MLP approximation', 'Training points', 'Location', 'Best');
xlabel('x');
ylabel('y');
title('Task 1: 1D Function Approximation with MLP');
grid on;

%  TASK 2: 2D SURFACE APPROXIMATION

fprintf(' TASK 2: 2D surface approximation \n');

% Now Creating a network with:
% 2 inputs: x1, x2
% 1 hidden layer (tanh)
% 1 linear output
% The target function is a smooth surface

% Generating 2D training data
N1 = 15;   % number of points along x1
N2 = 15;   % number of points along x2

x1 = linspace(0, 1, N1);
x2 = linspace(0, 1, N2);

[X1, X2] = meshgrid(x1, x2);  % N2 x N1 grid

% Defining target surface:
T2 = sin(pi * X1) .* cos(pi * X2);   % N2 x N1 matrix

% Flattening to vectors of samples
X1_vec = X1(:);    % (N1*N2 x 1)
X2_vec = X2(:);    % (N1*N2 x 1)
T2_vec = T2(:);    % (N1*N2 x 1)

N_2D = length(T2_vec);  % total number of training samples

% Network architecture for 2D
n_in_2D  = 2;
n_hid_2D = 8;     % hidden neurons
n_out_2D = 1;

% Initializing weights and biases
Wh_2D = 0.5 * randn(n_hid_2D, n_in_2D);   % (8x2)
bh_2D = 0.5 * randn(n_hid_2D, 1);         % (8x1)

Wo_2D = 0.5 * randn(n_out_2D, n_hid_2D);  % (1x8)
bo_2D = 0.5 * randn(n_out_2D, 1);         % (1x1)

% Training parameters
eta_2D      = 0.05;
n_epochs_2D = 3000;

E_hist_2D = zeros(n_epochs_2D, 1);

% Training loop for 2D (backpropagation)
for epoch = 1:n_epochs_2D
    sq_err_sum = 0;
    
    for i = 1:N_2D
        % Input vector (2x1)
        xi = [X1_vec(i); X2_vec(i)];
        ti = T2_vec(i);
        
        %  Forward pass 
        zh = Wh_2D * xi + bh_2D;
        ah = tanh(zh);               
        
        zo = Wo_2D * ah + bo_2D;     
        yi = zo;                     
        
        %  Error 
        err = yi - ti;
        sq_err_sum = sq_err_sum + 0.5*err^2;
        
        %  Backward pass 
        delta_out = err;             
        
        delta_hid = (Wo_2D' * delta_out) .* (1 - ah.^2);  
        
        %  Gradients 
        dWo = delta_out * ah';       
        dbo = delta_out;             
        
        dWh = delta_hid * xi';      
        dbh = delta_hid;             
        
        %  Update 
        Wo_2D = Wo_2D - eta_2D * dWo;
        bo_2D = bo_2D - eta_2D * dbo;
        
        Wh_2D = Wh_2D - eta_2D * dWh;
        bh_2D = bh_2D - eta_2D * dbh;
    end
    
    E_hist_2D(epoch) = sq_err_sum / N_2D;
    
    if mod(epoch, 500) == 0
        fprintf('2D Epoch %4d, MSE = %.6f\n', epoch, E_hist_2D(epoch));
    end
end

% Evaluating trained 2D network on grid
Y2_pred_vec = zeros(N_2D, 1);

for i = 1:N_2D
    xi = [X1_vec(i); X2_vec(i)];
    zh = Wh_2D * xi + bh_2D;
    ah = tanh(zh);
    zo = Wo_2D * ah + bo_2D;
    Y2_pred_vec(i) = zo;
end

Y2_pred = reshape(Y2_pred_vec, size(X1));  % reshaping it back to grid size

% Plots for 2D task

% Error vs epoch
figure;
plot(E_hist_2D, 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Mean squared error');
title('Task 2: 2D MLP – Training error (MSE)');
grid on;

% Target surface
figure;
surf(X1, X2, T2);
xlabel('x1'); ylabel('x2'); zlabel('t');
title('Task 2: Target surface');
shading interp; colorbar;

% Approximated surface
figure;
surf(X1, X2, Y2_pred);
xlabel('x1'); ylabel('x2'); zlabel('y_{MLP}');
title('Task 2: MLP-approximated surface');
shading interp; colorbar;

% Error surface (optional)
figure;
surf(X1, X2, T2 - Y2_pred);
xlabel('x1'); ylabel('x2'); zlabel('Error');
title('Task 2: Approximation error (target - MLP)');
shading interp; colorbar;
