function [xnew, signewsq, a] = backest(x, xold, sigsq, sigsqold);

%backward filter 
%variables

T = size(x,2);
a=0;
xnew(T)     = x(T);
signewsq(T) = sigsq(T);
for i = T-1 :-1: 2
   a(i)        = sigsq(i)/sigsqold(i+1);
   xnew(i)     = x(i) + a(i)*(xnew(i+1) - xold(i+1));
   signewsq(i) = sigsq(i) + a(i)*a(i)*(signewsq(i+1)-sigsqold(i+1));
end


