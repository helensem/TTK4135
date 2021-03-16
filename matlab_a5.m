x = 0:0.2:2*pi;
rng('default');
y = .1*(x + cos(x) + 1) + .1*rand(size(x));

N = size(x,2); % Number of data points

n_theta = 9;

% NN parameters
alpha = .2;
n_iter = 200;

% Allocate storage
ES = zeros(1,n_iter+1);
theta = zeros(n_theta,n_iter+1);

% initial value for parameters
theta(:,1) = zeros(n_theta,1);

% Initial value of E (objective function)
ES(1) = objfun(theta(:,1),x,y);

% function handle for objective function (for line search)
E = @(theta) objfun(theta,x,y);

% Initial BFGS matrix
I = eye(n_theta);
Hk = I;

% Initial gradient
dEdtheta = objfun_gradient(theta(:,1),x,y);

% Tolerance for convergence
tol = 1e-10;
 
k = 0;          % Iteration condition
exit = 0;       % Termination criteria
while not(exit),
    k = k+1;
    
    % BFGS search direction
    pk = - Hk*dEdtheta;
    
    % Linesearch update of theta -- TODO: You should update the linesearch function (see below)
    alpha = backtrackingLS(E,ES(k),theta(:,k),dEdtheta,pk);
    theta(:,k+1) = theta(:,k) + alpha*pk;

    % Calculate next Hk -- TODO: Update according to BFGS formula
    dEdtheta_new = objfun_gradient(theta(:,k+1),x,y);
    sk = theta(:,k+1) -theta(:,k); 
    yk = dEdtheta_new - dEdtheta;
    rhok = 1/(yk'*sk);
    if rhok > 0, % Do not update Hk if curvature condition is not fulfilled
        Hk = (I - rhok*sk*yk')*Hk*(I - rhok*yk*sk') + rhok*sk*sk';
    end
    % The reason for the if-clause is that we use a simple linesearch that
    % does not ensure that the curvature condition is fulfilled.
   
    % Next objective function value
    ES(k+1) = objfun(theta(:,k+1),x,y);
    
    % Assign for next iteration
    dEdtheta = dEdtheta_new;
    
    % Termination criteria
    exit = (abs(ES(k+1)-ES(k))<tol) | (k>=n_iter);
end

theta_opt = theta(:,k+1);

figure(1)
plot(1:k+1,ES(1:k+1));
xlabel('Iteration'); ylabel('E');

figure(2);
plot(x,y); hold on;
plot(x,f_NN(x,theta(:,k+1)),'c--'); hold off
xlabel('x'); ylabel('y');


function alpha = backtrackingLS(f,f_val,xk,df,pk)
% 
% backtracking line search (Algorithm 3.1 in N&W)
%
% f         handle for objective function
% f_val     objective function at xk
% xk        current iterate
% df        current gradient
% pk        search direction

alpha        = 1; % Initial guess at step length
rho          = .9; % < 1 reduction factor of alpha
c            = 1e-4; % Sufficient decrease constant

% TODO: fill in clause for while loop.
while (f(xk+alpha*pk)>f_val+c*alpha*df*pk'),
    alpha = rho*alpha;
    
    if alpha < 1e-8,
        error('alpha too small -- something is wrong');
    end
end

end

function E = objfun(theta,x,y)

N = size(x,2);
E = 0;
for i = 1:N,
    E = E + .5* (y(i) - f_NN(x(i),theta) )^2;
end

end

function dEdtheta = objfun_gradient(theta,x,y)    

N = size(x,2);

% Find gradient
dEdtheta = 0;
for i = 1:N,
     [dydthetai, hyi] = gradient_NN(x(i),theta);
     dEdtheta = dEdtheta - (y(i) - hyi)*dydthetai;
end

end

function y = f_NN(x,theta)
% Do a forward pass of a 1-2-1 Neural Network.
% 
%   x           input
%   theta       parameters theta = (w1, b1, w2, b2, w3, b3, w41, b41, w42, b4)
%   
%   y           output

% Activation function ("logistic")
sigma = @(r) 1./(1+exp(-r));

w1 = theta(1);
b1 = theta(2);
w2 = theta(3);
b2 = theta(4);
w3 = theta(5);
b3 = theta(6);
w41 = theta(7);
w42 = theta(8);
b4 = theta(9);

y1 = sigma(w1*x + b1);
y2 = sigma(w2*y1 + b2);
y3 = sigma(w3*y1 + b3);
y = sigma(w41*y2 + w42*y3 + b4);
end

function [dydtheta, y] = gradient_NN(x,theta)
% Calculate the gradient of the 1-2-1 NN wrt parameters.
% Note: This is an inefficient implementation, using "forward" (chain rule)
% differentiation. "Reverse" (mode of automatic differentiation) aka
% back propagation would be much more efficient, but the difference does 
% not matter for this small example.
% 
%   x           input
%   theta       parameters theta = (w1, b1, w2, b2, w3, b3, w41, w42, b4)
%   
%   y           output
%   dydtheta    gradient of y wrt theta

% Activation function ("logistic")
sigma = @(r) 1./(1+exp(-r));

% Derivative of activation function (expressed in terms of the output)
dsigmadr = @(sigma) sigma*(1-sigma);

w1 = theta(1);
b1 = theta(2);
w2 = theta(3);
b2 = theta(4);
w3 = theta(5);
b3 = theta(6);
w41 = theta(7);
w42 = theta(8);
b4 = theta(9);

y1 = sigma(w1*x + b1);
y2 = sigma(w2*y1 + b2);
y3 = sigma(w3*y1 + b3);
y = sigma(w41*y2 + w42*y3 + b4);

dydtheta = [
    dsigmadr(y)*w41*dsigmadr(y2)*w2*dsigmadr(y1)*x + dsigmadr(y)*w42*dsigmadr(y3)*w3*dsigmadr(y1)*x; % dydw1
    dsigmadr(y)*w41*dsigmadr(y2)*w2*dsigmadr(y1) + dsigmadr(y)*w42*dsigmadr(y3)*w3*dsigmadr(y1); % dydb1
    dsigmadr(y)*w41*dsigmadr(y2)*y1; % dydw2
    dsigmadr(y)*w41*dsigmadr(y2); % dydb2
    dsigmadr(y)*w42*dsigmadr(y3)*y1; % dydw3
    dsigmadr(y)*w42*dsigmadr(y3); % dydb3
    dsigmadr(y)*y2; % dydw41
    dsigmadr(y)*y3; % dydw42
    dsigmadr(y); % dydb4
    ];
    
end

