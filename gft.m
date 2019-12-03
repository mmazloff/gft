function [phat,yhat,CMEinv] = gft(x,y,kx,pvar,dvar,ifwd,varargin)
%-----------------------------------------------------------
% gft.m
% --------------------------------------------------
% needs function(s):
% --------------------------------------------------
% eiganal.m (only if idb > 2)
%-----------------------------------------------------------
% do forward or inverse fourier transform using least-squares:
%   generalized fourier analysis in N dimensions,
% where the data do not need to be regularly spaced, and
% the model parameters don't have to be regularly spaced OR
% band-limited.
% see gftndfwd.m for a version that only puts out the fwd problem matrix
%-----------------------------------------------------------
% it can operate in forward mode, in which it translates an input
% set of spectral amplitudes (pvar)
% into an output time series (or space series) (yhat)
% or in inverse mode, in which case
%  it takes an input time series (y) and
% estimates the spectral parameters (phat).
%  (it also produces an output series made from them (yhat))
%-----------------------------------------------------------
% when doing the inverse, can use vector or scalar pvar and dvar
%   if length(dvar) == 1 it assumes that the data all have
%   identically distributed independent errors
% if length(dvar) == n_data, then it is data error var
% also,
% The model parameters can have different expected sizes, which is
% communicated through input vector "pvar", which is a scalar if all
% parameters have the same expected magnitudes.
%-----------------------------------------------------------
% the following quirk has been removed:  It's in backup/gft_080822.m
% !! caution: for backward compatibility, if pvar is scalar,
%    then it is not used, and dvar is assumed to be divided by
%    model error variance. as dvar/pvar
% it was removed because it made it complicated to figure out
%  what the output error estimate was.
%-----------------------------------------------------------
% gft can invert either underdetermined (invert in data space)
%  or overdetermined (invert in model space)
% and solve the matrix problem or invert the matrix.
%-----------------------------------------------------------
% parameter ifwd
% Allows a choice between real transform (real data -> sines and cosines)
% and complex transform.  The real transform saves time for real data,
% since the number of unknowns is halved, since they would be
% conjugate-symmetric around 0 frequency.
% the complex inversion part may not be exactly right yet,
%  with conjugate transposes..
%-----------------------------------------------------------
% to do:
% make it more efficient; see syk/rad_cov_homog.m for
% faster ways to weight the matrices.
% files that call this subprogram
%/net/lau/bdc/matlab% grep "gft(" *.m
%  fit_gft.m
%  mk_fake_radial_cov.m:%[phat,yhat] = gft(dx,ycov,kx,p,snr,ifwd)
% testfft.m:[a2 phat] = gft(t',p',omega',snr);
% xbtsectdiff.m:    [ampl tmphat] = gft(yearday',tmp,omega,pin,snr,1);
%/net/lau/bdc/matlab% grep "gft(" */*.m
% mpl/chk_freq.m:%[phat,yhat] = gft(datasig.t_ax,data,omega,p,snr,ifwd);
% mpl/chk_freq2.m:%[phat,yhat] = gft(datasig.t_ax,data,omega,p,snr,ifwd);
%-----------------------------------------------------------
% for 2-d (space-time) problems, see specfit2d.m
%-----------------------------------------------------------
% bdc 4/30/02; revised 8/19/08
%-----------------------------------------------------------
% input (expect x, kx to be column vectors,
%   y can have more than one column if all have the same x )
% ***I have wanted to make this able to do multiple x's with multiple kx's
%  for multiple y's, but this has not been done.  For now, if have
%  multiple x's or kx's, have to loop outside this program.
% x = abscissa (independent variable) (N x 1)
%     (N is the dimension of x-space and kx-space)
% y = ordinate (dependent variable) (n_data x Ny)
%    (can have more than one column);
%      each column is an independent realization of the process, and will
%      be fit.  There can be Ny columns, all assumed to have the same x
%    if ifwd = 0 (or 2), y is not  used..
% kx =  wavenumbers to fit to (n_freq x 1) (= 2*pi/wavelength)
%    if ifwd < 2 (real), then there are 2 parameters per element of kx;
%    for the sine and cosine.
%    if ifwd >= 2 (complex), then kx had better have + and - frequencies
%     in it, or the data cannot be fit properly!!
% p = input spectral amplitudes if ifwd = 0 or 2.
%     if ifwd = 1 or 3, then
%       if p is n_par long, this is a prior for the model and
%        dvar (see below) is the prior for the data error;
%       if p is NOT n_par long, p is not used, and prior is
%       assumed to be I; in other words, dvar is p/(data var).
% dvar = vector or scalar data noise VARIANCE
%    (small dvar means try to fit the data tightly)
%    if this is a scalar, it applies to all data
%    **If dvar is a vector corresponding to the number of data
%    then the vector dvar is used as a varying dvar for different
%    data.
%    for more complicated situations, see other programs.
%    **Note that if have many different data series (multiple columns of y),
%    with possibly different dvar, this program has to treat them all
%    the same!
% ifwd = switch between forward and inverse transform.
%    (!!and real or complex)  This means it does 2 things at once; sorry!
%    "just do fwd problem" means transforming from an input vector of
%    amplitudes to a time (or space) series.
%    see variable i_cos for cosine or sine transform. (forces symmetry
%     or anti-symmetry about the origin.)
%
%   0,2 = just do forward problem: ignore y, transform input p to yhat.
%         (no need for inverse!)
%   1,3 = do transform: ignore p, input y is inv. transf. to phat (and yhat)
%    0 = real fwd problem (sines and cosines), 1 = real inverse,
%    2 = complex fwd (exp(-ikx*x)), 3 = complex inverse.
%
% output:
% phat = matrix of estimated parameters; one column for each column of y
%    columns are n_par = 2*n_freq long if are doing sines and cosines.
%    first have n_freq cosine amplitudes, then n_freq sine amplitudes.
%    if complex, n_par = n_freq (again, assumes you have put neg freqs in kx)
%    if ifwd == 0, then phat = p.
% yhat = estimated values of y
% CMEinv = output posterior model parameter uncertainty (error) covariance
%--------------------------------------

%--------------------------------------
% assumes the forward problem:
%--------------------------------------
% complex version:
% (ifwd > 1)
%--------------------------------------
%          n_freq
%   y(x) = sum [phat(j)*exp(i*kx(j)*x)]
%          j=1
%--------------------------------------
% so the output phat() vector is complex and n_par (=n_freq) long
%--------------------------------------
% OR:
%--------------------------------------
% real version:
% (ifwd < 2)
% ***assumes real input
% Just sines and cosines
%--------------------------------------
%          n_freq
%   y(x) = sum [phat(j)*cos(kx(j)*x) + phat(j+n_freq)*sin(kx(j)*x) ]
%          j=1
%--------------------------------------
% so the output phat() vector is real and n_par = 2*n_freq long
%--------------------------------------
%----------------------------------------
if (exist('varargin','var') & ~isempty(varargin))
  % Read varargin and parse it into variable = value expressions
  % expect pairs on input:  'variable',value
  % loop over all pairs to set values
  for ido = 1:floor(length(varargin)/2)
    eval([varargin{2*ido-1} ' = varargin{2*ido};'])
  end
end
%----------------------------------------
% check dimensions of input
% number of data = number of rows in x
[n_data nd_x] = size(x);
% number of unknowns = number of rows in kx
[n_freq nd_k] = size(kx);

if (nd_x > 1 | nd_k > 1)
  disp('x and kx must be column vectors in gft.m')
  disp('keyboard in gft.m')
  keyboard
end

% for the future; not yet enabled
if (nd_x ~= nd_k)
  disp('dimensions of x, kx do not match in gft.m')
  size(x)
  size(kx)
  size(y)
  disp('keyboard in gft.m')
  keyboard
end

% number of parameters = number of rows in pvar
% (otherwise it is assumed to be scalar)
[n_parp nd_p] = size(pvar);
if (nd_p > 1)
  disp('pvar must be a column vector in gft.m')
  disp('keyboard in gft.m')
  keyboard
end

%--------------------------------------
%--------------------------------------
% need to make fwd problem matrices
% (for generalizations to more dimensions, see tide3dfwd.m)
% (or specfit2d.m)  (or lsft.m)
%--------------------------------------
disp('make fwd problem')
% n_par = number of unknowns
tic
  if (ifwd < 2)
    % are real; just sines and cosines
    %--------------------------------------
    if ~exist('i_cos','var') || isempty(i_cos)
      % i_cos = 1 means do cosine transform, -1 = sine
      % (only applies when ifwd < 2) (real)
      i_cos = 0;
    end
    if (i_cos == 0)
      n_par = 2*n_freq;
      %-----------------------------------------------------------
      % use the outer product to make a matrix with
      % n_data rows and n_par columns
      % fwd problem matrix
      G=[cos(x*kx') sin(x*kx')];
    elseif (i_cos == 1)
      disp('cosine transform')
      n_par = n_freq;
      G=[cos(x*kx')];
    elseif (i_cos == -1)
      disp('sine transform')
      n_par = n_freq;
      G=[sin(x*kx')];
    end
  else
    % complex
    % fwd problem matrix
    disp('assume + and - frequencies are given, or input is real')
    n_par = n_freq;
    % if + and - frequencies are given
    G=exp(i*x*kx');
    % need + and - frequencies to fit general input series
    %disp('convert to + and - frequencies')
    %n_par = 2*n_freq;
    %G=[exp(i*x*kx') exp(-i*x*kx')];
  end  
  %size(G)
toc

%--------------------------------------
% could add other parameters (see tide3d.m)

%--------------------------------------
if (ifwd == 0 | ifwd == 2)
  % just convert harmonic coefficients in pvar to y-hat
  if (n_parp ~= n_par)
    disp('pvar has wrong number of rows in gft.m')
    keyboard
  end
  yhat = G*pvar;
  phat = pvar;
  return
end
%--------------------------------------
% stay here for inverse
%--------------------------------------
if ~exist('idb','var') || isempty(idb)
  % choose debugging printout (hardwired)
  idb = 0;
  %idb = 1;
  % if idb > 0, it will plot intermediate results
  % if idb > 2, it will list out the eigenvalues and make a plot
end
disp(['ido= ' num2str(idb)'])

if ~exist('evralmin','var') || isempty(evrelmin)
% set min ev ratio for the analysis; only used if idb > 0
evrelmin = 1.e-2;
end

%--------------------------------------
% number of data series = number of columns in y
[n_data2 nd_y] = size(y);
if (n_data ~= n_data2)
  disp('dimensions of x, y do not match in gft.m')
  keyboard
end

%--------------------------------------
% see if noise variance is a scalar or vector
n_dvar = length(dvar);
% sometimes need inverse noise variance, not noise variance
dvarinv = 1./dvar;

pvarinv = pvar.^-1;

%--------------------------------------
% inverts in either data or model space,
% depending on which is smaller
% or whether error is desired
%--------------------------------------
if ~exist('imk_err','var') || isempty(imk_err)
  % imk_err = 1 means make model space error covariance
  % (forces overdetermined inverse)
  imk_err = 0;
end
%--------------------------------------
%if (n_par <= n_data)
if ~exist('i_od','var') || isempty(i_od)
  % added feature: i_od = 1 forces overdetermined inverse
  %i_od = 1;
  i_od = 0;
end
%--------------------------------------
if (i_od == 0 & imk_err == 1)
  %disp('need cme; force OD inverse')
  i_od = 1;
end
if (i_od == 1)
  disp('forcing od inverse')
end

if (n_par < n_data | i_od == 1)
  disp('do overdetermined inverse')
  % overdetermined inverse (more data than unknowns)
  % invert in model space.
  % R = dvar is either a scalar or a vector n_data long
  % estimate is:
  % phat = (G'*R^-1*G + P^-1)^-1 * G'*R^-1*y
  % if noise var is constant (Rn) (R = Rn * I)
  % AND if signal variance is constant (Ps) (P = Ps * I)
  % phat = (Rn^-1 *(G'*G + I*Rn/Ps))^-1 * Rn^-1 * G'*y
  % phat = (G'*G + I*Rn/Ps)^-1 * G'*y
  % otherwise, if signal variance is constant (Ps) (P = Ps * I)
  % phat = (Ps^-1 *(Ps*G'*R^-1*G + I))^-1 * G'*R^-1*y
  % phat = (Ps*G'*R^-1*G + I)^-1 * Ps * G'*R^-1*y
  % phat = (Ps*G'*R^-1*G + I)^-1 * Ps*G'*R^-1*y
  % since Ps is a scalar, just carry through:
  % phat = (G'*(Ps*R^-1)*G + I)^-1 * G'*(Ps*R^-1)*y
  % phat = (G'*(Ps*R^-1)*G + I)^-1 * G'*(Ps*R^-1)*y
  % so just need the SNR in either case
  
  disp('make backprojected weighted data')
  tic
  % wtd = G'*R^-1*y;

  if (n_dvar == 1)
    disp('have scalar dvar: constant data error')
    % fix up for when only have one data error variance
    wtd = dvarinv(1)*G'*y;
  elseif (n_dvar == n_data)
    disp('have vector dvar; different for each dataum')
    % make G'*(R^-1)*y
    % first: weighted G-transpose (need it later)
    wtGt = G'*diag(dvarinv);
    % weighted data
    wtd = wtGt*y;
  else
    disp('illegal length for dvarinv in gft.m; should be 1 or n_par')
  end
  toc
  % (wtd is column vector n_par long)
  % (not any more: will have as many columns as y)
  if (idb > 0)
    figure(1)
    plot(wtd)
    keyboard
  end
  
  disp('make matrix to invert')
  % G'*R^-1*G + P^-1
  %size(G)
  tic
  if (n_dvar == 1)
    disp('have scalar dvar')
    %size(diag(ones(2*n_freq,1)*dvar))
    if (n_parp ~= n_par)
      disp('identity is prior for parameters (pvar(1) only used)')
      % noise and signal are both constant (R = Rn * I)
      % invert (G'*G + I*Rn/Ps)
      %CME = G'*G + diag(ones(n_par,1)/snr);
      %CME = G'*G + diag(repmat(snr,n_par,1));
      %if (ifwd < 2)
        % real
        CME = dvarinv*G'*G + pvarinv(1)*eye(n_par);
      %else
      %  complex:  This probably needs a conj?
      %  CME = dvarinv*G'*G + pvarinv(1)*eye(n_par);
      %end
    else
      disp('use pvar as prior for parameters')
      %if (ifwd < 2)
        % real
        CME = dvarinv*G'*G + diag(pvarinv);
      %else
      %  CME = dvarinv*G'*G + diag(pvarinv);
      %end
    end
  elseif (n_dvar == n_data)
    % have vector snr for data
    if (n_parp ~= n_par)
      disp('identity is prior for parameters (p not used)')
      % R^(-1) is already multiplied by P
      %if (ifwd < 2)
        CME = wtGt*G + pvarinv(1)*eye(n_par);
      %else
      %  CME = wtGt*G + pvarinv(1)*eye(n_par);
      %end
    else
      disp('use p as prior for parameters')
      %if (ifwd < 2)
        CME = wtGt*G + diag(pvarinv);
      %else
      %  CME = wtGt*G + diag(pvarinv);
      %end
    end
  else
    disp('illegal length for snr')
  end
  toc
  if (idb > 0)
    figure(2)
    imagesc(CME)
    colorbar
    keyboard
  end
  % optional: diagnostics
  % for eigenvalue / eigenvector analysis and plotting,
  % use function eiganal.m
  % set minimum significance level for plotting
  if (idb > 2)
    [ev] = eiganal(CME,evrelmin);
  end  
  disp('do the estimate of the parameters')
  tic
  if (nd_y < n_par & imk_err == 0)
    % solve each one (uses cholesky factorization)
    phat = CME\wtd;
  else
    % make inverse operator and multiply
    if (imk_err == 0)
      phat = inv(CME)*wtd;
    else
      disp('make inverse CME')
      CMEinv = inv(CME);
      phat = CMEinv*wtd;
    end
  end
  toc
  if (idb > 0)
    figure(3)
    plot(phat);
    title('phat')
    keyboard
  end
else
  disp('do underdetermined inverse')
  %--------------------------------------
  % matlab unwtd least squares
  %Z=G\d;
  %
  %--------------------------------------
  % wtd least squares:
  % estimate is:
  % phat = P*G' * (G*P*G' + R)^-1 * y
  % if noise var is constant (Rn) (R = Rn * I)
  % AND if signal variance is constant (Ps) (P = Ps * I)
  %      = Ps*G' * (G*Ps*G' + Rs*I)^-1 * y
  %      = G' * (G*G' + I*Rn/Ps)^-1 * y
  %--------------------------------------------

  disp('make data data covariance matrix')
  %Rdd=G*Rmm*G'+Rnn;
  %Rdd=Rmm * (G*G'+Rnn/Rmm);
  % scalar signal and noise
  %Rdd=G*G'+ diag(ones(n_data,1)/snr);
  %Rdd=G*G'+ diag(repmat(1/snr(1),n_data,1));
  tic
  if (n_dvar == 1)
    disp('have scalar dvar')
    if (n_parp ~= n_par)
      disp('have scalar pvar')
      disp('identity is prior for parameters')
      Rdd = pvar(1)*G*G' + dvar*eye(n_data);
    else
      disp('use pvar as prior for parameters')
      pGt = diag(pvar)*G';
      Rdd = G*pGt + dvar*eye(n_data);
    end
  elseif (n_dvar == n_data)
    disp('have data error')
    if (n_parp ~= n_par)
      disp('have scalar pvar')
      disp('identity is prior for parameters')
      Rdd = pvar(1)*G*G' + diag(dvar);
    else
      disp('use pvar as prior for parameters')
      pGt = diag(pvar)*G';
      Rdd = G*pGt + diag(dvar);
    end
  else
    disp('illegal length for snr')
  end
  toc
  % optional: diagnostics
  % for eigenvalue / eigenvector analysis and plotting,
  % use function eiganal.m
  % set minimum significance level for listing and plotting
  %evrelmin = 1.e-2;
  if (idb > 2)
    [ev] = eiganal(Rdd,evrelmin);
  end
  disp('make inverse operator')
  tic
  %L=Rmm*G'/Rdd;
  if (n_parp ~= n_par)
    % (the Rmm's cancel out..)
    L=G'/Rdd;
  else
    L=pGt/Rdd;
  end
  toc
  disp('make estimate')
  tic
  phat = L*y;
  toc
end

% phat = parameter estimate

% estimated data
yhat=G*phat;

if (idb > 0)
  figure(4)
  plot(x,y,x,yhat)
  title('y and yhat')
  keyboard
end

% choose whether or not to continue
iresid = 1;
if (iresid == 0)
  return
end
%--------------------------------------------
% look at solution and residuals: see wpolyfit.m
%--------------------------------------------
iplot = 0;
if (iplot == 1)
 plot(x,yhat,':',x,y,'-')
 title('original and reconstructed data')
end

% difference ("residuals")
% caution: this may be a matrix, with each input time series a column
diff=y-yhat;

if (iplot == 1)
  % plot residuals vs section range (to look for structure)
  plot(x,diff,'x',x,y,'+');
  title('data and residuals')
end

% normalized residuals
%wtdr=diff./(diag(Rnn).^.5);
% normalized model parameters
%wtdm=m./(diag(Rmm).^.5);

% sum variance before and after
% variance before and after
%dvar0 = norm(y);
%dvar1 = norm(diff);
% note that this sums over BOTH dimensions, so if y has
% multiple time series, this sums over all time series.
if (ifwd < 2)
  dvar0 = sum(sum(y.*y));
  dvar1 = sum(sum(diff.*diff));
else
  dvar0 = sum(sum(y.*conj(y)));
  dvar1 = sum(sum(diff.*conj(diff)));
end

format short g
disp('[dvar0 dvar1 dvar1/dvar0]')
disp([dvar0 dvar1 dvar1/dvar0])
format

% R= resolution
%R=L*G; 
% diagonal of resolution matrix
%diR=diag(R);
% plot(diR);
%trR=trace(R)

% error in mode amplitudes
% if no resolution for modes= error = Rmm
%Error=Rmm-R*Rmm';

% error of each mode; not worrying about non-orthogonality
% of modes (off diagonal elements of "Error")
%Error1=(diag(Error)).^.5;

% for plotting error bars
%e1=m-Error1;
%e2=m+Error1;

% convert to physical space errors
%error=G*Error*G';                       

%--------------------------------------------
