clear; clc; close all;

% Generate Training Data 
x = 0.1:1/22:1;                                     % 20 input samples
d = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2;  % Target function

% Initialize Parameters 
% Hidden layer weights (5 neurons, layer 1)
w11_1 = rand(1);   % input -> hidden neuron1
w12_1 = rand(1);   
w13_1 = rand(1);   
w14_1 = rand(1);   
w15_1 = rand(1);   

% Hidden layer biases
b1_1  = rand(1);   
b2_1  = rand(1);  
b3_1  = rand(1);   
b4_1  = rand(1);   
b5_1  = rand(1);   

% Output layer weights (from 5 hidden neurons to output, layer 2)
w11_2 = rand(1);   % hidden1 -> output
w12_2 = rand(1);   
w13_2 = rand(1);   
w14_2 = rand(1);   
w15_2 = rand(1);  

% Output layer bias
b1_2  = rand(1);   % output neuron bias

eta = 0.1;         % learning rate


% Training Loop 
for ep = 1:1000000
    for i = 1:length(x)

        % Feedforward 
        % Hidden layer net inputs
        v1_1 = x(i)*w11_1 + b1_1;
        v2_1 = x(i)*w12_1 + b2_1;
        v3_1 = x(i)*w13_1 + b3_1;
        v4_1 = x(i)*w14_1 + b4_1;
        v5_1 = x(i)*w15_1 + b5_1;

        % Hidden layer activations (sigmoid)
        y1_1 = 1/(1 + exp(-v1_1));
        y2_1 = 1/(1 + exp(-v2_1));
        y3_1 = 1/(1 + exp(-v3_1));
        y4_1 = 1/(1 + exp(-v4_1));
        y5_1 = 1/(1 + exp(-v5_1));

        % Output layer
        v1_2 = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + y5_1*w15_2 + b1_2;
        y1_2 = v1_2;      % linear activation

        % Error 
        e = d(i) - y1_2;

        % Backpropagation
        % Output layer delta
        delta1_2 = e;

        % Hidden layer deltas (sigmoid derivative: y*(1-y))
        delta1_1 = y1_1*(1 - y1_1) * delta1_2 * w11_2;
        delta2_1 = y2_1*(1 - y2_1) * delta1_2 * w12_2;
        delta3_1 = y3_1*(1 - y3_1) * delta1_2 * w13_2;
        delta4_1 = y4_1*(1 - y4_1) * delta1_2 * w14_2;
        delta5_1 = y5_1*(1 - y5_1) * delta1_2 * w15_2;

        % Weight Updates 
        % Output layer
        w11_2 = w11_2 + eta*delta1_2*y1_1;
        w12_2 = w12_2 + eta*delta1_2*y2_1;
        w13_2 = w13_2 + eta*delta1_2*y3_1;
        w14_2 = w14_2 + eta*delta1_2*y4_1;
        w15_2 = w15_2 + eta*delta1_2*y5_1;
        b1_2  = b1_2  + eta*delta1_2;

        % Hidden layer
        w11_1 = w11_1 + eta*delta1_1*x(i);
        w12_1 = w12_1 + eta*delta2_1*x(i);
        w13_1 = w13_1 + eta*delta3_1*x(i);
        w14_1 = w14_1 + eta*delta4_1*x(i);
        w15_1 = w15_1 + eta*delta5_1*x(i);

        b1_1  = b1_1 + eta*delta1_1;
        b2_1  = b2_1 + eta*delta2_1;
        b3_1  = b3_1 + eta*delta3_1;
        b4_1  = b4_1 + eta*delta4_1;
        b5_1  = b5_1 + eta*delta5_1;
    end
end


% Testing 

x_test = 0.1:1/220:1;    % denser test grid
y_test = (1 + 0.6*sin(2*pi*x_test/0.7) + 0.3*sin(2*pi*x_test))/2;
Y_pred = zeros(1,length(x_test));

for i = 1:length(x_test)
    v1_1 = x_test(i)*w11_1 + b1_1;
    v2_1 = x_test(i)*w12_1 + b2_1;
    v3_1 = x_test(i)*w13_1 + b3_1;
    v4_1 = x_test(i)*w14_1 + b4_1;
    v5_1 = x_test(i)*w15_1 + b5_1;

    y1_1 = 1/(1 + exp(-v1_1));
    y2_1 = 1/(1 + exp(-v2_1));
    y3_1 = 1/(1 + exp(-v3_1));
    y4_1 = 1/(1 + exp(-v4_1));
    y5_1 = 1/(1 + exp(-v5_1));

    Y_pred(i) = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + y5_1*w15_2 + b1_2;
end


% Visualization 
figure(1);

plot(x,d,'bo','MarkerSize',8,'LineWidth',1.5); hold on
plot(x_test,y_test,'b--','LineWidth',2);
plot(x_test,Y_pred,'r','LineWidth',2);
legend('Training Data','Target Function','MLP Approximation','Location','best');
title('Function Approximation Using MLP (5 Sigmoid Neurons)'); 
xlabel('x'); ylabel('y');



% 2D Surface Approximation

% Training Data
x1 = 0:0.1:1;
x2 = 0:0.1:1;
[X1, X2] = meshgrid(x1, x2);

T = sin(pi*X1).*cos(pi*X2);   % target surface

% Flatten grid to training vectors
x1v = X1(:)';
x2v = X2(:)';
dv  = T(:)';
N   = length(dv);

% Initialize Weights (5 hidden neurons)
w11 = rand(1); w21 = rand(1); b1 = rand(1);
w12 = rand(1); w22 = rand(1); b2 = rand(1);
w13 = rand(1); w23 = rand(1); b3 = rand(1);
w14 = rand(1); w24 = rand(1); b4 = rand(1);
w15 = rand(1); w25 = rand(1); b5 = rand(1);

% Output layer
wo1 = rand(1); wo2 = rand(1); wo3 = rand(1); wo4 = rand(1); wo5 = rand(1);
bo  = rand(1);

eta = 0.05;     % learning rate

%  Training Loop 
for ep = 1:30000      % training epochs
    for k = 1:N

        x1k = x1v(k);
        x2k = x2v(k);

        % Hidden neurons
        v1 = x1k*w11 + x2k*w21 + b1;   y1 = 1/(1+exp(-v1));
        v2 = x1k*w12 + x2k*w22 + b2;   y2 = 1/(1+exp(-v2));
        v3 = x1k*w13 + x2k*w23 + b3;   y3 = 1/(1+exp(-v3));
        v4 = x1k*w14 + x2k*w24 + b4;   y4 = 1/(1+exp(-v4));
        v5 = x1k*w15 + x2k*w25 + b5;   y5 = 1/(1+exp(-v5));

        % Output neuron
        vout = y1*wo1 + y2*wo2 + y3*wo3 + y4*wo4 + y5*wo5 + bo;
        yout = vout;

        % Error
        e = dv(k) - yout;

        % Output delta
        d_out = e;

        % Hidden deltas (sigmoid derivative)
        d1 = y1*(1-y1)*d_out*wo1;
        d2 = y2*(1-y2)*d_out*wo2;
        d3 = y3*(1-y3)*d_out*wo3;
        d4 = y4*(1-y4)*d_out*wo4;
        d5 = y5*(1-y5)*d_out*wo5;

        % Update output weights
        wo1 = wo1 + eta*d_out*y1;
        wo2 = wo2 + eta*d_out*y2;
        wo3 = wo3 + eta*d_out*y3;
        wo4 = wo4 + eta*d_out*y4;
        wo5 = wo5 + eta*d_out*y5;
        bo  = bo  + eta*d_out;

        % Update hidden weights (each neuron has 2 input weights)
        % Neuron 1
        w11 = w11 + eta*d1*x1k;   w21 = w21 + eta*d1*x2k;   b1 = b1 + eta*d1;
        % Neuron 2
        w12 = w12 + eta*d2*x1k;   w22 = w22 + eta*d2*x2k;   b2 = b2 + eta*d2;
        % Neuron 3
        w13 = w13 + eta*d3*x1k;   w23 = w23 + eta*d3*x2k;   b3 = b3 + eta*d3;
        % Neuron 4
        w14 = w14 + eta*d4*x1k;   w24 = w24 + eta*d4*x2k;   b4 = b4 + eta*d4;
        % Neuron 5
        w15 = w15 + eta*d5*x1k;   w25 = w25 + eta*d5*x2k;   b5 = b5 + eta*d5;

    end
end

% Surface Reconstruction 
Y = zeros(size(X1));

for r = 1:size(X1,1)
    for c = 1:size(X1,2)

        xx1 = X1(r,c);
        xx2 = X2(r,c);

        v1 = xx1*w11 + xx2*w21 + b1;   y1 = 1/(1+exp(-v1));
        v2 = xx1*w12 + xx2*w22 + b2;   y2 = 1/(1+exp(-v2));
        v3 = xx1*w13 + xx2*w23 + b3;   y3 = 1/(1+exp(-v3));
        v4 = xx1*w14 + xx2*w24 + b4;   y4 = 1/(1+exp(-v4));
        v5 = xx1*w15 + xx2*w25 + b5;   y5 = 1/(1+exp(-v5));

        Y(r,c) = y1*wo1 + y2*wo2 + y3*wo3 + y4*wo4 + y5*wo5 + bo;
    end
end

figure(2);
surf(X1, X2, T);
title('Target Surface  z = sin(\pi x_1)cos(\pi x_2)');
xlabel('x_1'); ylabel('x_2'); zlabel('z'); shading interp;

figure(3);
surf(X1, X2, Y);
title('MLP Approximation (2D Surface)');
xlabel('x_1'); ylabel('x_2'); zlabel('y_{MLP}'); shading interp;
