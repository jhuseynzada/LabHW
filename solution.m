clear; clc; close all;

data = readmatrix('Data.txt');   % works for comma-separated file

x1 = data(:,1);
x2 = data(:,2);
d  = data(:,3);                  % desired outputs: -1 or 1

X = [x1 x2];                     % N x 2 feature matrix
N = size(X,1);

rng('shuffle');          % random seed
w1 = randn(1);
w2 = randn(1);
b  = randn(1);

eta        = 0.1;        % learning rate (0 < eta < 1)
max_epochs = 100;        % safety limit


%  Perceptron training loop
for epoch = 1:max_epochs
    num_errors = 0;

    for n = 1:N
        x1n = X(n,1);
        x2n = X(n,2);

        % Current output of the perceptron
        v = x1n*w1 + x2n*w2 + b;
        if v > 0
            y = 1;
        else
            y = -1;
        end

        % Error for this example
        e = d(n) - y;

        % Updating weights and bias if error is non-zero
        if e ~= 0
            w1 = w1 + eta*e*x1n;
            w2 = w2 + eta*e*x2n;
            b  = b  + eta*e;
            num_errors = num_errors + 1;
        end
    end

    fprintf('Epoch %d, errors: %d\n', epoch, num_errors);

    % If there were no misclassifications , -> we are done
    if num_errors == 0
        fprintf('Training converged after %d epochs.\n', epoch);
        break;
    end
end

%  Evaluate perceptron on training data

y_perc = zeros(N,1);

for n = 1:N
    v = X(n,1)*w1 + X(n,2)*w2 + b;
    if v > 0
        y_perc(n) = 1;
    else
        y_perc(n) = -1;
    end
end

perc_accuracy = mean(y_perc == d) * 100;
fprintf('Perceptron training accuracy: %.2f %%\n', perc_accuracy);


%  Plot training data and decision boundary (optional but nice for report)
figure; hold on; grid on; box on;
title('Perceptron classification and decision boundary');
xlabel('x_1'); ylabel('x_2');

% Plot class -1 and +1 with different markers
idx_neg = (d == -1);
idx_pos = (d ==  1);

plot(X(idx_neg,1), X(idx_neg,2), 'ro', 'DisplayName','Class -1');
plot(X(idx_pos,1), X(idx_pos,2), 'b+', 'DisplayName','Class +1');

% Decision boundary: w1*x1 + w2*x2 + b = 0
% => x2 = (-b - w1*x1)/w2
x1_min = min(X(:,1)) - 0.5;
x1_max = max(X(:,1)) + 0.5;
xx1 = linspace(x1_min, x1_max, 100);

if abs(w2) > 1e-6
    xx2 = (-b - w1*xx1) / w2;
    plot(xx1, xx2, 'k-', 'LineWidth', 2, 'DisplayName','Decision boundary');
end

legend('Location','best');
hold off;
