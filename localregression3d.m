function f = localregression3d(w,x0,y0,z0,degree,kernel,h,wts)
%
% function f = localregression3d(w,x0,y0,z0,degree,kernel,h,wts)
%
% <x>,<y>,<z>,<w> are matrices of the same size with the data
% <x0>,<y0>,<z0> are matrices of the same size with the points to evaluate at
% <degree> (optional) is 0 or 1.  default: 1.
% <kernel> (optional) is 'epan'.  default: 'epan'.
% <h> is the bandwidth [xb yb zb].  values can be Inf.
%   can be a scalar in which case we use that for all three dimensions.
% <wts> (optional) is a matrix the same size as <w> with non-negative numbers.
%   these are weights that are applied to the local regression in order to
%   allow certain points to have more influence than others.  note that
%   the weights enter the regression in exactly the same way as the kernel
%   weights.  default: ones(size(<w>)).
%
% We require that there be no NaNs in <w> and <wts>.
%
% return a matrix with the value of the function at <x0>,<y0>,<z0>.
%
% singular warnings are suppressed.  can return NaNs.
% note that entries with NaN in <w> are ignored.
%
% see also localregression.m and localregression4d.m.
%
% example:
% [x0,y0,z0] = ndgrid(-1:.1:1);
% w0 = localregression3d(w,flatten(x0),flatten(y0),flatten(z0));
% w0actual = flatten(sin(x0) + cos(y0) + tan(z0));
% figure;
% scatter(w0,w0actual,'r.');
% axissquarify;
% xlabel('local regression fit'); ylabel('true values');
%
% Copyright ???? Kendrick Kay <knk@stanford.edu>
% Modified slightly by Bob Dougherty <bobd@syanford.edu>

% input
if ~exist('degree','var') || isempty(degree)
  degree = 1;
end
if ~exist('kernel','var') || isempty(kernel)
  kernel = 'epan';
end
if ~exist('wts','var') || isempty(wts)
  wts = ones(size(w));
  wtsopt = 1;
else
  wtsopt = 0;
end
if length(h)==1
  h = repmat(h,[1 3]);
end

% prep
nx = size(w,1);
ny = size(w,2);
nz = size(w,3);

warning off;

% do it
f = NaN*zeros(size(x0));
for pp=1:numel(x0)

    % calculate k and ix
    % figure out where the subvolume is
    indices = {max(1,ceil(x0(pp)-h(1))):min(nx,floor(x0(pp)+h(1))) ...
               max(1,ceil(y0(pp)-h(2))):min(ny,floor(y0(pp)+h(2))) ...
               max(1,ceil(z0(pp)-h(3))):min(nz,floor(z0(pp)+h(3)))};
    ix = false(nx,ny,nz);
    ix(indices{:}) = true;  % this is a logical matrix that will return the subvolume elements

    % calculate kernel weights
    temp = bsxfun(@plus,reshape(((indices{1} - x0(pp))/h(1)).^2,[],1), ...
                        reshape(((indices{2} - y0(pp))/h(2)).^2,1,[]));
    temp = bsxfun(@plus,temp, ...
                        reshape(((indices{3} - z0(pp))/h(3)).^2,1,1,[]));
    k = 0.75*(1-temp);
    k(k<0) = 0;
    k = k(:);  % o x 1 (length is number of elements in subvolume)

    % get out early
    if isempty(k)
        continue;
    end

    % filter out
    numx = indices{1}(end)-indices{1}(1)+1;
    numy = indices{2}(end)-indices{2}(1)+1;
    numz = indices{3}(end)-indices{3}(1)+1;
    xA = repmat(reshape(indices{1},[numx 1]),[1 numy numz]);
    xA = xA(:);
    yA = repmat(reshape(indices{2},[1 numy]),[numx 1 numz]);
    yA = yA(:);
    zA = repmat(reshape(indices{3},[1 1 numz]),[numx numy 1]);
    zA = zA(:);
    wA = w(ix);
    if wtsopt
        wtsA = ones(length(xA),1);
    else
        wtsA = wts(ix);
    end
    n = length(xA);

    % form X matrices
    if degree==0
        X = ones(n,1);
        x0X = [1];
    else
        X = [xA yA zA ones(n,1)];
        x0X = [x0(pp) y0(pp) z0(pp) 1];
    end

    % AVOID THIS FOR SPEED REASONS
    %  % form W matrix
    %  W = diag(k);

    % solve it
    k = k .* wtsA;
    sol = (X'*(repmat(k,[1 size(X,2)]).*X)) \ (X'*(k.*wA));  % LIKE THIS FOR SPEED
    if isfinite(sol)
        f(pp) = x0X*sol;
    end
end

return
