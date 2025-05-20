function aberrated_opto = generate_optotype(C, optotype, VA, Dpupil, var, age, phi)
    % generate_optotype - Generates an optotype image convolved with an eye model
    % that includes optical aberrations, age-related degradation, and neural filtering.
    %
    % Inputs:
    %   - C: Zernike coefficient matrix (first column contains Zernike indices)
    %   - optotype: character to render as the optotype
    %   - VA: visual acuity value in decimal units
    %   - Dpupil: pupil diameter in millimeters
    %   - var: noise variance to be added in the spatial domain
    %   - age: age of the subject, used to adjust modulation transfer
    %   - phi: scaling parameter for contrast sensitivity function
    %
    % Output:
    %   - aberrated_opto: simulated optotype image after optical, retinal and neural processing
    %
    % This function:
    %   1. Constructs the wavefront using Zernike polynomials and pupil geometry.
    %   2. Computes the point spread function (PSF) of the eye.
    %   3. Renders the optotype at the desired visual acuity and convolves it with the PSF.
    %   4. Simulates cone sampling on the retina.
    %   5. Calculates Spatial Contrast Sensitivity Function (SCSF) and Mean Optical Transfer Function (MOTF).
    %   6. Constructs the neural transfer function (NTF = SCSF / MOTF).
    %   7. Applies the NTF in the frequency domain.
    %   8. Adds spatial Gaussian noise.
 
    addpath(genpath('TracerLibrary'))
    
    % Default data:
    ConeFrequency = 2.0;
    rad2arc = 60*180/pi;
    
    % Coefficients passed with single index
    K = C(:,1);
    % Leaving just coefficient values in C
    C(:,1) = [];
    
    % Wavelength
    lambda = 570*1e-6;
    
    % Linear dimensions in mm
    C = C*1e-3;
    Rp = Dpupil/2;
    
    % Pupil window size (PUPILws) in wavelengths.
    % This size and the number of points determine the interval in the
    % transformed space, deltak = 1/PUPILws, kmax - kmin = N/PUPILws;
    % If we define the input in terms of wavelengths, the output is given in
    % radians, so if PUPILws = 50000 lambda, deltak = 1/50000 rad = 0.02 mrad =
    % 0.0688 arcmin. 
    NX = 100;
    Rpw = Rp/lambda;
    dxw = Rpw/20;
    PUPILws = 40*Rpw;
    % IMPORTANT: make the selection so that PUPILws/dxw is an even integer
    % IMPORTANT: The size of the PSF window in radians is 1/dxw, the pixel size
    % of the PSF window is 1/PUPILws radians. 
    
    xw = -PUPILws/2:dxw:PUPILws/2;
    yw = xw;
    [xxw,yyw] = meshgrid(xw, yw);
    rrw2 = xxw.^2 + yyw.^2;
    I = rrw2 <= Rpw(1).^2;
    
    % Wavefront
    W = zeros(size(xxw));
    W(I) = ZernikePolynomials(xxw(I)/Rpw(1),...
                              yyw(I)/Rpw(1), K, 'osa')*C;
    
    % Pupil transmitance
    % Assumed standard Styles-Crawford effect (SCE), rho = 0.12.
    %    R. A. Applegate and V. Lakshminarayanan, "Parametric representation of
    %    Stiles–Crawford functions: normal variation of peak location and
    %    directionality," J.Opt.Soc.Am.A 10, 1611–1623 (1993).
    % Use rho = 0 to remove SCE
    rho = 0.12*lambda^2; 
    P = zeros(size(xxw));
    I = rrw2 <= Rpw(end).^2;
    P(I) = exp(-(rho/2)*rrw2(I)); 
    
    % Normalization (pupil area in wavelengths squared)
    Norm = pi*Rpw(end)^2;
    
    % Complex pupil
    P2 = (1/Norm)*P.*exp(-(1i*2*pi/lambda)*W);
    
    % -------------------------------- PSF --------------------------------
    PSF = fft2(P2);
    PSF = fftshift(PSF);
    PSF = PSF.*conj(PSF);
    PSF = PSF/sum(PSF(:));
    
    % Output coordinates for the PSF (in radians)
    du = 1/PUPILws;
    PSFws = 1/dxw;
    u = -PSFws/2:du:PSFws/2;
    u = u*rad2arc;
    
    % arcmin corresponding to one pixel in the PSF, 
    du = u(2) - u(1); % this is equivalent to rad2arc/PUPILws
    % pixels per arcmin: round(MAR/du)5
    ppam = NX*VA/5;
    duimg = 1/ppam;
    
    % Trim the PSF image to keep only the majority of the signal. This makes
    % the later convolution slightly faster
    jj = (length(PSF)-1)/2+1;
    Cs2 = PSF(jj, jj:end);
    j2 = find(cumsum(Cs2)/sum(Cs2) > 0.95, 1, 'first');
    
    if j2 >= jj
        j2 = jj-1;
    end

    PSF2 = PSF(jj-j2:jj+j2, jj-j2:jj+j2);
    PSF2 = imresize(PSF2, du / duimg); %This one has duimg pixel size
    
    % ------------------------ SIMULATE OPTOTYPE ------------------------
    optotype = [double(optotype) 0];
    % Foregrond and background colors
    fC = [255,255,255]; 
    bC = [0,0,0];
    Contrast = 1;
    % Bitmap of the optotype5.
    I = renderTextFT(optotype, bC, fC, 'fonts/sloan.ttf', 5/VA*ppam);
    I = (Contrast/255)*double(rgb2gray(I));
    
    % ---------------------- CONVOLUTION WITH PSF ----------------------
    I2 = imresize(I, 1);
    PSF2 = PSF2 / sum(PSF2(:));
    Inb = convolve2(I2, PSF2);
    Nx1 = size(Inb);
    Nb = 150;
    I = zeros(Nx1 + Nb);
    I(Nb/2:(Nb/2 + Nx1(1) - 1), Nb/2:(Nb/2 + Nx1(2) - 1)) = Inb;
    
    % ------------------------- RETINAL SAMPLING -------------------------
    du = duimg;
    duCone = 1/ConeFrequency;
    dvCone = duCone*sqrt(3)/2;
    [N,M] = size(I);
    ulim = (M-1)*du/2;
    vlim = (N-1)*du/2;
    uCone = -ulim:duCone:ulim;
    vCone = -vlim:dvCone:vlim;
    [uuCone, vvCone] = meshgrid(uCone, vCone);
    uuCone(2:2:end,:) = uuCone(2:2:end,:) + duCone/2;
    
    u = -ulim:du:ulim;
    v = -vlim:du:vlim;
    [uu,vv] = meshgrid(u,v);
    IatCones = griddata(uu, vv, I, uuCone, vvCone, 'linear');
    I = griddata(uuCone, vvCone, IatCones, uu, vv, 'linear');
    I(isnan(I)) = 0;
    I = 1-I;

    % ------------------------------- SCSF -------------------------------
    % Parameters from table 5 of https://jov.arvojournals.org/article.aspx?articleid=2121870
    n = 100;
    n1 = size(I,1);
    r = Dpupil/2;
    f0 = 4.1726;
    f1 = 1.3625;
    a = 0.8493;
    p = 0.7786;
    Gain = 373.08;
    
    % Extend 1D SCSF to 2D for application in frequency domain

    % Polar coordinates
    if rem(n,2) == 0
        X0 = -floor(n/2)+0.5:floor(n/2)-0.5;
    else
        X0 = -floor(n/2):floor(n/2);
    end
    
    X = repmat(X0,[n,1]);
    Y = repmat(X0',[1,n]);
    
    % Retinal pixe size (mm) = (pixel size*focal length)/(pupil diameter*refractive index)
    pixel_size = (570*1e-6*22.24)/(2*r*1.336);  % mm
    X = X.*pixel_size;
    Y = Y.*pixel_size;
    [theta, rho] = cart2pol(X, Y);  % rho in mm
    
    lpmm = 0.5/pixel_size;
    cicl_deg = (22.24*lpmm/1.336)*(pi/180);
    factor = cicl_deg/lpmm;  % lppmAcdeg
    
    rho = lpmm.*rho;
    
    SCSF = Gain.*(sech((rho./(phi*f0*factor)).^p) - a.*sech(rho./(phi*f1*factor)));
    SCSF = SCSF/max(max(SCSF));
    
    % ------------------------------- MOTF -------------------------------
    D = 70;  % normalization factor to 70 years
    ageFactor = 1+(age/D)^4;
    MOTF = 1/(1+ageFactor/7)*(0.426*exp((-0.028*factor).*rho)+0.574*exp((-0.37*factor).*rho))+1/(1+7/ageFactor)*(0.125*exp((-37*factor).*rho)+0.877*exp((-360*factor).*rho));
    MOTF = MOTF/max(max(MOTF));
    
    % -------------------------------- NTF --------------------------------
    NTF = SCSF./MOTF;
    NTF = NTF/max(max(NTF));
    pad = floor((n1-n)/2);
    NTF = padarray(NTF, [pad pad], 'both');
    if n1-size(NTF, 1) == 1
        NTF = padarray(NTF, [1 1], 'replicate', 'post');
    end
    

    % ---------------------- APPLY NTF IN FREQUENCY ----------------------
    I_TF = fft2(fftshift(I));
    freq_opto = I_TF.*fftshift(NTF);
    freq_opto_shift = fftshift(freq_opto);
    aberrated_opto = real(ifftshift(ifft2(ifftshift(freq_opto_shift))));

    % ---------------------- ADD NOISE IN REAL SPACE ----------------------
    N = var*randn(size(aberrated_opto));
    aberrated_opto = aberrated_opto+N;

end
