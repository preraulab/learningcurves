function binplot(varargin)

if nargin==1
    varargin{1}=logical(N);
    times=1:length(N);
elseif nargin==2
    times=varargin{1};
    N=varargin{2};
end

oinds=N==1;
xinds=N==0;

hold on
stem(times(oinds),ones(1,sum(oinds)),'color', [0 .5 0],'marker','none','linewidth',2)
stem(times(xinds),-ones(1,sum(xinds)),'color', [1 0 0],'marker','none','linewidth',2)

ylim([-1.1 1.1]);
set(gca,'ytick',[-.5 .5],'yticklabel',{'Incorrect','Correct'});