% figure
% format shorte
% 
% data=load('sol.dat')
% 
% N=size(data,2);
% x=linspace(0+1/N,1,N+1);
% xx=x(2:end);
% 
% plot(xx,data)
% xlabel('x');
% ylabel('u');

% ------------for the 1d advection-diffusion-reaction 3-variable problem, 
%             comment the code above and use below
figure
format shorte

data=load('sol.dat');

N=size(data,2);
x=linspace(0,1,N+1);
xx=x(2:end);

plot(xx,data)
xlabel('x');
ylabel('u');

% --------- compute the L2 and L-max error by comparing the solution with 
%           the reference solution (generated using PIROCK rtol=atol=10^-7

refData = load('ref_sol.dat'); % Load the reference solution
N_ref=size(refData,2);

if (N~=N_ref)
    error('The number of data points in the solution and reference do not match.');
end
x_ref=linspace(0,1,N_ref+1);
xx_ref=x_ref(2:end);

l2err  = 0.0;
lmaxerr= 0.0;
for i = 1:length(xx)
    abserr = abs(xx(i) - xx_ref(i));
    if (abserr>lmaxerr)
        lmaxerr = abserr;
    end
    l2err = l2err + abserr.^2;
end
fprintf("L2-error: %.14e\n", l2err)
fprintf("L-mas: %.14e\n", lmaxerr)