"""Reproduce the three conditional physical examples of edition 2.

All generated observations are synthetic. Noise levels and apparatus values
are specified assumptions, not experimental measurements. The three models
are independent examples; no unified new gravitational dynamics is asserted.
Run: python revision2/code/physics_revision.py
Requires numpy scipy matplotlib.
"""
from pathlib import Path
import json
from decimal import Decimal, localcontext
import numpy as np
from scipy.constants import c, G, h, hbar, elementary_charge
from scipy.optimize import least_squares, brentq
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'; FIG=ROOT/'figures'
RESULTS.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})
TEAL='#2f6672'; ORANGE='#bd653b'; GRAY='#7e888d'


def savefig(fig,name):
    fig.tight_layout()
    fig.savefig(FIG/(name+'.png'),dpi=240,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def gauss_segments(bounds,n):
    x,w=leggauss(n)
    xx=[];ww=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        if b>a:
            xx.append((b-a)*x/2+(a+b)/2);ww.append(w*(b-a)/2)
    return np.concatenate(xx),np.concatenate(ww)


DISK={'radius_m':5e-3,'thickness_m':100e-6,'density_kg_m3':19300.,
      'gap_m':50e-6,'gap_modulation_m':.1e-6,'drive_Hz':100.,
      'assumed_harmonic_force_standard_error_N':3e-20}


def disk_force_and_gradient(gap,lam=None,order=32):
    """Exact finite-cylinder volume integral reduced to two quadratures.

    Both disks are uniform, coaxial and equal. lam=None is Newton's kernel;
    finite lam is a unit-strength Yukawa correction. Returns signed dF/dgap
    and positive attractive force. No infinite-plane approximation is used.
    The Yukawa lateral integral is truncated at 40 lambda only when that is
    smaller than the diameter; exponential suppression bounds its relevance.
    """
    a,t,rho=DISK['radius_m'],DISK['thickness_m'],DISK['density_kg_m3']
    ww,wweight=gauss_segments([0,t,2*t],order)
    vertical_weight=np.minimum(ww,2*t-ww)*wweight
    z=gap+ww
    if lam is None:
        rb=sorted(set([0.,gap/4,gap,4*gap,16*gap,a,2*a]))
        rb=[x for x in rb if x<=2*a]
    else:
        cutoff=min(2*a,40*lam)
        rb=sorted(set([0.]+[min(cutoff,lam*x) for x in [.1,.5,1,2,5,10,20,40]]+[cutoff]))
    rr,rweight=gauss_segments(rb,order)
    overlap=2*a*a*np.arccos(np.clip(rr/(2*a),-1,1))-.5*rr*np.sqrt(np.maximum(0,4*a*a-rr*rr))
    s=np.sqrt(rr[:,None]**2+z[None,:]**2)
    zz=z[None,:]
    if lam is None:
        force_kernel=zz/s**3
        derivative=(1-3*zz**2/s**2)/s**3
    else:
        exp=np.exp(-s/lam)
        force_kernel=zz*(1+s/lam)*exp/s**3
        derivative=exp*((1+s/lam)*(1-3*zz**2/s**2)-zz**2/lam**2)/s**3
    weight=(rr*rweight*overlap)[:,None]*vertical_weight[None,:]
    norm=2*np.pi*G*rho*rho
    return float(norm*np.sum(weight*force_kernel)),float(norm*np.sum(weight*derivative))


def compact_harmonic(R,beta=1.,gap=None,order=32):
    gap=DISK['gap_m'] if gap is None else gap
    count=max(1,int(np.ceil(38*R/gap)))
    gradient=sum(2*beta*disk_force_and_gradient(gap,R/n,order)[1] for n in range(1,count+1))
    return -DISK['gap_modulation_m']*gradient


def point_displacement_response(omega_R_over_c,r_over_R=3.,tower=True):
    x=np.asarray(omega_R_over_c,dtype=float)
    modes=np.arange(0,100 if tower else 2)
    kr=np.sqrt((x[...,None]+0j)**2-modes[None,:]**2)*r_over_R
    weights=np.ones_like(modes,dtype=float);weights[1:]=2
    return np.sum(weights*np.exp(1j*kr)*(2-2j*kr-kr*kr),axis=-1)


def compact_example():
    R=10e-6
    fnewt,gn= disk_force_and_gradient(DISK['gap_m'],None,48)
    f1=compact_harmonic(R,order=40)
    f1hi=compact_harmonic(R,order=64)
    fnewthi,gnhi=disk_force_and_gradient(DISK['gap_m'],None,72)
    sigma=DISK['assumed_harmonic_force_standard_error_N']
    first=-DISK['gap_modulation_m']*2*disk_force_and_gradient(DISK['gap_m'],R,48)[1]
    Rs=np.geomspace(1e-6,100e-6,70)
    responses=np.array([compact_harmonic(r) for r in Rs])
    firsts=np.array([-DISK['gap_modulation_m']*2*disk_force_and_gradient(DISK['gap_m'],r)[1] for r in Rs])
    b95=1.959963984540054*sigma/responses
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    axes[0].loglog(Rs*1e6,responses,label='Вся башня мод',color=TEAL,lw=2)
    axes[0].loglog(Rs*1e6,firsts,'--',label='Только первая пара',color=ORANGE)
    axes[0].axhline(1.959963984540054*sigma,color=GRAY,ls=':',label='1,96 σF по условию')
    axes[0].set(xlabel='Радиус R, мкм',ylabel='Гармоника добавочной силы, Н',ylim=(1e-24,1e-14))
    axes[0].legend(fontsize=8)
    axes[1].loglog(Rs*1e6,b95,color=TEAL,lw=2)
    axes[1].fill_between(Rs*1e6,b95,1,where=b95<1,color=TEAL,alpha=.13)
    axes[1].set(xlabel='Радиус R, мкм',ylabel='Условный порог β при 1,96 σF',ylim=(1e-5,1))
    axes[1].text(.04,.06,'Условная чувствительность\nпри заданной σF\nНе экспериментальное исключение',transform=axes[1].transAxes,fontsize=8)
    for ax in axes:ax.grid(alpha=.15,which='both')
    savefig(fig,'physics_compact_force')

    # Ordinary one-mediator competitor fitted at two distances, then tested
    # at other distances. This synthetic comparison is not a uniqueness proof.
    trueR=20e-6;truebeta=.1
    gaps=np.array([20,30,40,60,100])*1e-6
    targets=np.array([compact_harmonic(trueR,truebeta,z) for z in gaps])
    train=np.array([3,4])
    def residual(v):
        alpha,lam=np.exp(v)
        prediction=np.array([-DISK['gap_modulation_m']*alpha*disk_force_and_gradient(z,lam)[1] for z in gaps[train]])
        return (prediction-targets[train])/targets[train]
    fit=least_squares(residual,np.log([.2,20e-6]),xtol=1e-11,ftol=1e-11,gtol=1e-11)
    alpha,lam=np.exp(fit.x)
    pred=np.array([-DISK['gap_modulation_m']*alpha*disk_force_and_gradient(z,lam)[1] for z in gaps])
    xx=np.linspace(0,3.5,1001)
    full=point_displacement_response(xx)
    single=point_displacement_response(xx,tower=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    axes[0].plot(xx,np.abs(full),color=TEAL,label='Компактная башня')
    axes[0].plot(xx,np.abs(single),'--',color=ORANGE,label='Нулевая мода и одна пара')
    for n in [1,2,3]:axes[0].axvline(n,color=GRAY,ls=':',lw=.8)
    axes[0].set(xlabel='ωR / c',ylabel='Модуль отклика на смещение',title='Точечная модель при r = 3R')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.15)
    axes[1].plot(gaps*1e6,(targets-pred)/sigma,'o-',color=TEAL)
    axes[1].scatter(gaps[train]*1e6,((targets-pred)/sigma)[train],s=75,facecolors='none',edgecolors=ORANGE,label='Точки подгонки')
    axes[1].axhline(0,color=GRAY,lw=.8)
    axes[1].set(xlabel='Зазор, мкм',ylabel='Различие сил / заданная σF',title='Башня и один посредник')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.15)
    savefig(fig,'physics_compact_dynamic')
    return {'status':'conditional calculation; no new force observation',
       'apparatus':DISK,'R_m':R,'beta':1.,'extra_force_harmonic_N':f1,
       'first_mode_force_harmonic_N':first,'fraction_from_higher_modes':float((f1-first)/f1),
       'Newton_force_N':fnewt,'Newton_force_harmonic_N':-DISK['gap_modulation_m']*gn,
       'quadrature_relative_change_extra':abs(f1hi-f1)/f1hi,
       'quadrature_relative_change_Newton_harmonic':abs(gnhi-gn)/abs(gnhi),
       'beta_nominal_95_sensitivity_at_R':float(1.959963984540054*sigma/f1),
       'first_mode_energy_eV':float(hbar*c/R/elementary_charge),
       'first_mode_threshold_Hz':float(c/(2*np.pi*R)),
       'mechanical_drive_omegaR_over_c':float(2*np.pi*DISK['drive_Hz']*R/c),
       'scalar_vs_single_mediator':{'R_m':trueR,'beta':truebeta,'gaps_m':gaps.tolist(),
          'training_indices':train.tolist(),'fitted_alpha':float(alpha),'fitted_lambda_m':float(lam),
          'compact_harmonics_N':targets.tolist(),'single_harmonics_N':pred.tolist(),
          'difference_in_assumed_sigma':((targets-pred)/sigma).tolist()},
       'curve_R_m':Rs.tolist(),'curve_beta95':b95.tolist()}


def clock_matrix(A,ell,d,tau,f,orientation=1):
    S0=4*A*A*tau/(1+(2*np.pi*f*tau)**2)
    sy=2*S0*(1-np.exp(-d*d/(2*ell*ell)))
    sa=c**4*S0/(ell*ell)
    cross=-orientation*c*c*S0*d/(ell*ell)*np.exp(-d*d/(8*ell*ell))
    return np.array([[sy,cross],[cross,sa]],dtype=complex)


def clock_example():
    cfg={'A':1e-20,'ell_m':1000.,'d_m':1000.,'tau_s':10.,'frequency_Hz':.01,
         'clock_ASD_per_sqrt_Hz':1e-18,'acceleration_ASD_m_s2_per_sqrt_Hz':1e-7,
         'clock_averaging_s':1.,'gravimeter_lowpass_Hz':.1,
         'common_noise_correlation':.3,'independent_spectral_samples_per_orientation':4096,
         'seed':20261113}
    A,ell,d,tau,f=[cfg[x] for x in ['A','ell_m','d_m','tau_s','frequency_Hz']]
    ny,na=cfg['clock_ASD_per_sqrt_Hz'],cfg['acceleration_ASD_m_s2_per_sqrt_Hz']
    H=np.array([np.sinc(f*cfg['clock_averaging_s']),1/(1-1j*f/cfg['gravimeter_lowpass_Hz'])])
    N=np.array([[ny*ny,.3*ny*na],[.3*ny*na,na*na]],complex)
    scale=np.array([ny,na]);M=cfg['independent_spectral_samples_per_orientation']
    rng=np.random.default_rng(cfg['seed'])
    matrices=[];estimates=[]
    for orientation in [1,-1]:
        field=clock_matrix(A,ell,d,tau,f,orientation)
        S=H[:,None]*field*H[None,:].conj()+N
        Sn=S/scale[:,None]/scale[None,:]
        L=np.linalg.cholesky(Sn)
        z=(rng.normal(size=(M,2))+1j*rng.normal(size=(M,2)))/np.sqrt(2)
        obs=z@L.T
        est=obs.T@obs.conj()/M
        matrices.append(S);estimates.append(est*scale[:,None]*scale[None,:])
    odd=(estimates[0][0,1]-estimates[1][0,1])/2
    even=(estimates[0][0,1]+estimates[1][0,1])/2
    template=H[0]*H[1].conj()*clock_matrix(1.,ell,d,tau,f)[0,1]
    angle=np.angle(template)
    variance=sum((S[0,0].real*S[1,1].real+np.real(np.exp(-2j*angle)*S[0,1]**2))/(2*M) for S in matrices)/4
    qhat=float(np.real(odd/template));qse=float(np.sqrt(variance)/abs(template))
    real_cross_se=np.sqrt(sum((S[0,0].real*S[1,1].real+np.real(S[0,1]**2))/(2*M) for S in matrices)/4)
    null_estimates=[]
    nullL=np.linalg.cholesky(N/scale[:,None]/scale[None,:])
    for orientation in [1,-1]:
        z=(rng.normal(size=(M,2))+1j*rng.normal(size=(M,2)))/np.sqrt(2)
        obs=z@nullL.T
        null_estimates.append((obs.T@obs.conj()/M)*scale[:,None]*scale[None,:])
    null_odd=(null_estimates[0][0,1]-null_estimates[1][0,1])/2
    null_even=(null_estimates[0][0,1]+null_estimates[1][0,1])/2
    null_qhat=float(np.real(null_odd/template))
    null_variance=(N[0,0].real*N[1,1].real+np.real(np.exp(-2j*angle)*N[0,1]**2))/(4*M)
    null_qse=float(np.sqrt(null_variance)/abs(template))
    def expected_snr(amplitude):
        mats=[H[:,None]*clock_matrix(amplitude,ell,d,tau,f,s)*H[None,:].conj()+N for s in [1,-1]]
        var=sum((S[0,0].real*S[1,1].real+np.real(np.exp(-2j*angle)*S[0,1]**2))/(2*M) for S in mats)/4
        return float(amplitude*amplitude*abs(template)/np.sqrt(var))
    targetA=10**brentq(lambda z:expected_snr(10**z)-5,-25,-16)
    field=clock_matrix(A,ell,d,tau,f)
    coh=abs(field[0,1])**2/(field[0,0].real*field[1,1].real)
    obscoh=abs(matrices[0][0,1])**2/(matrices[0][0,0].real*matrices[0][1,1].real)
    xx=np.linspace(.03,6,400);u=xx*xx/4
    gamma=u/np.sinh(u)
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    ax=axes[0];ax.set_xlim(-2.1,2.1);ax.set_ylim(-.8,1)
    ax.plot([-1.5,1.5],[0,0],color=GRAY,lw=1.5)
    ax.scatter([-1.5,1.5],[0,0],s=180,color=TEAL)
    ax.scatter([0],[0],s=150,color=ORANGE,marker='s')
    ax.text(-1.5,.28,'Часы b',ha='center');ax.text(1.5,.28,'Часы a',ha='center')
    ax.text(0,-.34,'Гравиметр\naₓ вдоль базы',ha='center')
    ax.annotate('',xy=(.65,.12),xytext=(.1,.12),arrowprops={'arrowstyle':'->','color':ORANGE})
    ax.annotate('',xy=(1.5,.65),xytext=(-1.5,.65),arrowprops={'arrowstyle':'<->','color':GRAY})
    ax.text(0,.73,'База d',ha='center');ax.axis('off')
    axes[1].plot(xx,gamma,color=TEAL,label='Идеальная когерентность γ²')
    axes[1].plot(xx,1-np.exp(-xx*xx/2),'--',color=ORANGE,label='Дисперсия часов / (2A²)')
    axes[1].set(xlabel='d / ℓ',ylabel='Безразмерная величина',ylim=(0,1.05))
    axes[1].grid(alpha=.15);axes[1].legend(fontsize=8,loc='center right',bbox_to_anchor=(1,.72))
    savefig(fig,'physics_clock_geometry')

    freqs=np.geomspace(.001,1,250)
    sf=np.array([clock_matrix(A,ell,d,tau,x) for x in freqs])
    amps=np.geomspace(1e-23,1e-18,100);snrs=np.array([expected_snr(a) for a in amps])
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    factors=1e25
    desired=[(H[0]*H[1].conj()*field[0,1]).real*factors,N[0,1].real*factors]
    measured=[odd.real*factors,even.real*factors]
    axes[0].bar(np.array([0,1])-.17,desired,.34,color=TEAL,label='Ожидание модели')
    axes[0].bar(np.array([0,1])+.17,measured,.34,color=ORANGE,label='Синтетическая оценка',yerr=real_cross_se*factors,capsize=3)
    axes[0].set_xticks([0,1],['Нечётная часть\nполевой сигнал','Чётная часть\nобщая помеха'])
    axes[0].set_ylabel('Re взаимного спектра × 10²⁵, м/с²/Гц')
    axes[0].axhline(0,color=GRAY,lw=.8);axes[0].legend(fontsize=8)
    axes[1].loglog(amps,snrs,color=TEAL)
    axes[1].axhline(5,color=GRAY,ls=':');axes[1].scatter([targetA],[5],color=ORANGE)
    axes[1].set(xlabel='Амплитуда A',ylabel='Ожидаемое отношение к ошибке',ylim=(.01,1000))
    axes[1].grid(alpha=.15,which='both')
    savefig(fig,'physics_clock_control')
    return {'status':'synthetic Fourier samples; supplied noise PSD and known geometry',
      'parameters':cfg,'Fourier_convention':'time amplitudes multiply exp(-i 2 pi f t); S_ab=E[X_a X_b*]',
      'Hy':[float(H[0].real),float(H[0].imag)],'Ha':[float(H[1].real),float(H[1].imag)],
      'field_Syy':float(field[0,0].real),'field_Saa':float(field[1,1].real),
      'field_Sya':float(field[0,1].real),'ideal_coherence_squared':float(coh),
      'observed_coherence_squared_orientation_plus':float(obscoh),
      'synthetic_odd_cross_real':float(odd.real),'synthetic_odd_cross_imag':float(odd.imag),
      'expected_odd_cross_real':float((template*A*A).real),'expected_odd_cross_imag':float((template*A*A).imag),
      'synthetic_even_cross_real':float(even.real),'expected_even_cross_real':float(N[0,1].real),
      'A_squared_estimate':qhat,'A_squared_standard_error':qse,
      'A_estimate':float(np.sqrt(max(0,qhat))),
      'nominal_A_squared_interval_2sigma':[qhat-2*qse,qhat+2*qse],
      'real_cross_standard_error':float(real_cross_se),
      'common_noise_only_control':{'A_squared_estimate':null_qhat,'A_squared_standard_error':null_qse,
           'estimate_in_standard_errors':null_qhat/null_qse,
           'even_cross_real':float(null_even.real),'odd_cross_real':float(null_odd.real)},
      'expected_SNR_at_injected_A':expected_snr(A),'A_for_expected_SNR5':targetA,
      'minimum_total_time_days_if_effective_bandwidth_0p001Hz':2*M/.001/86400,
      'scope':'The interval uses supplied covariances, fixed ell/tau and Gaussian spectral sampling; it is not a global empirical false-alarm probability or a fit of all field parameters.',
      'rindler_check':{'lapse':'N(x)=1+a0 x/c^2; N_second=0; a_proper=c^2 N_prime/N',
          'a0_m_s2':9.81,'baseline_m':1.,'redshift_first_order':float(9.81/c**2),
          'curvature':'Riemann tensor is identically zero; explicit Rindler-to-Minkowski coordinate transform'},
      'formal_Newtonian_density_rms_kg_m3':float(c*c*np.sqrt(15)*A/(4*np.pi*G*ell*ell))}


def arm_transfer(f,L,mu):
    Omega=2*np.pi*np.asarray(f)*L/c
    return .5*(np.exp(1j*Omega*(3+mu)/2)*np.sinc(Omega*(1-mu)/(2*np.pi))+
               np.exp(1j*Omega*(1+mu)/2)*np.sinc(Omega*(1+mu)/(2*np.pi)))


def tidal_curvature_example():
    """One static, prescribed lapse with a nonzero tidal component.

    This is a local geometric/control example, not a sourced Einstein model.
    High-precision arithmetic is needed only to avoid cancellation of the
    O(1e-19) clock contrast when lapse values near one are subtracted.
    It does not represent the precision of a physical frequency measurement.
    The uncertainty budget is first-order propagation of declared independent
    clock/accelerometer errors and two distinct geometry calibration errors.
    """
    D=Decimal
    with localcontext() as ctx:
        ctx.prec=70
        cc=D(299792458);c2=cc*cc
        a0=D('9.81');kappa=D('3e-6');half=D('100')
        sy=D('1e-20');sa=D('2e-5');sh=D('0.001');sb=D('0.00001')
        def lapse(x,k=kappa):
            return 1+a0*x/c2+k*x*x/(2*c2)
        def alpha(x,k=kappa):
            return (a0+k*x)/lapse(x,k)
        def alpha_prime(x):
            n=lapse(x)
            return kappa/n-(a0+kappa*x)**2/(c2*n*n)
        np_,nm=lapse(half),lapse(-half)
        yp,ym=np_-1,nm-1
        contrast=yp+ym
        analytical_contrast=kappa*half*half/c2
        clock_kappa=c2*contrast/(half*half)
        gradient=(alpha(half)-alpha(-half))/(2*half)
        kinetic=alpha(D(0))**2/c2
        corrected_gradient=gradient+kinetic
        # The two clock differences share the central clock: cov = sigma_y^2.
        sigma_contrast=6**D('0.5')*sy
        clock_noise=c2*sigma_contrast/(half*half)
        clock_half=2*kappa*sh/half
        clock_center=2*a0*sb/(half*half)
        clock_total=(clock_noise**2+clock_half**2+clock_center**2).sqrt()
        grad_noise=(sa*sa/(2*half*half)+(2*a0*sa/c2)**2).sqrt()
        grad_half=abs(gradient)*sh/half
        grad_center=abs((alpha_prime(half)-alpha_prime(-half))/(2*half))*sb
        grad_total=(grad_noise**2+grad_half**2+grad_center**2).sqrt()
        r_gradient=(alpha(half,D(0))-alpha(-half,D(0)))/(2*half)
        r_corrected=r_gradient+kinetic
        result={
            'status':'prescribed static local metric; conditional measurement budget, not a sourced or observed gravity model',
            'lapse':'N(x)=1+a0*x/c^2+kappa*x^2/(2*c^2)',
            'Riemann_convention':'R^rho_(sigma mu nu)=d_mu Gamma^rho_(nu sigma)-d_nu Gamma^rho_(mu sigma)+Gamma^rho_(mu lambda)Gamma^lambda_(nu sigma)-Gamma^rho_(nu lambda)Gamma^lambda_(mu sigma)',
            'measured_tidal_combination':'K=c^2*R_hat0hatxhat0hatx=alpha_prime+alpha^2/c^2=c^2*N_second/N',
            'parameters':{'a0_m_s2':float(a0),'kappa_s_minus2':float(kappa),'half_baseline_m':float(half),
                          'clock_fractional_standard_error_each':float(sy),
                          'acceleration_standard_error_each_m_s2':float(sa),
                          'common_half_baseline_standard_error_m':float(sh),
                          'midpoint_offset_standard_error_m':float(sb)},
            'lapse_plus_decimal':str(np_),'lapse_minus_decimal':str(nm),
            'clock_plus_fractional_shift':float(yp),'clock_minus_fractional_shift':float(ym),
            'even_clock_contrast':float(contrast),'analytic_even_clock_contrast':float(analytical_contrast),
            'decimal_clock_identity_absolute_error':float(abs(contrast-analytical_contrast)),
            'direct_float64_lapse_contrast':float(np_)+float(nm)-2.,
            'clock_difference_noise_covariance':[[float(2*sy*sy),float(sy*sy)],[float(sy*sy),float(2*sy*sy)]],
            'even_clock_contrast_noise_standard_error':float(sigma_contrast),
            'central_curvature_m_minus2':float(kappa/c2),'kappa_from_clocks_s_minus2':float(clock_kappa),
            'clock_kappa_noise_error_s_minus2':float(clock_noise),
            'clock_kappa_half_baseline_error_s_minus2':float(clock_half),
            'clock_kappa_centering_error_s_minus2':float(clock_center),
            'clock_kappa_total_standard_error_s_minus2':float(clock_total),
            'clock_signal_to_total_standard_error':float(kappa/clock_total),
            'alpha_plus_m_s2':float(alpha(half)),'alpha_minus_m_s2':float(alpha(-half)),
            'support_acceleration_difference_m_s2':float(alpha(half)-alpha(-half)),
            'support_acceleration_gradient_s_minus2':float(gradient),
            'kinematic_correction_s_minus2':float(kinetic),
            'corrected_gradient_s_minus2':float(corrected_gradient),
            'finite_baseline_error_s_minus2':float(corrected_gradient-kappa),
            'gradient_kappa_noise_error_s_minus2':float(grad_noise),
            'gradient_kappa_half_baseline_error_s_minus2':float(grad_half),
            'gradient_kappa_centering_error_s_minus2':float(grad_center),
            'gradient_kappa_total_standard_error_s_minus2':float(grad_total),
            'gradient_signal_to_total_standard_error':float(kappa/grad_total),
            'local_free_fall_relative_acceleration_m_s2':float(-2*kappa*half),
            'rindler_corrected_finite_gradient_s_minus2':float(r_corrected),
            'rindler_exact_curvature_m_minus2':0.,'rindler_exact_even_clock_contrast':0.,
            'decimal_precision':70,
            'scope':'Static worldlines and calibrated frequency transfer; a shared central clock creates correlated differences. The half-baseline and midpoint errors are separate. Free-fall interpretation uses a nonrotating local inertial frame. No Einstein source, laboratory performance or global significance is claimed.'}
    return result


def michelson_transfer(f,L,theta=0.,azimuth=0.):
    n=np.array([np.sin(theta)*np.cos(azimuth),np.sin(theta)*np.sin(azimuth),np.cos(theta)])
    p=np.array([np.cos(theta)*np.cos(azimuth),np.cos(theta)*np.sin(azimuth),-np.sin(theta)])
    q=np.array([-np.sin(azimuth),np.cos(azimuth),0.])
    e=np.outer(p,p)-np.outer(q,q)
    return .5*(e[0,0]*arm_transfer(f,L,n[0])-e[1,1]*arm_transfer(f,L,n[1]))


def interferometer_example():
    cfg={'L_m':40.,'laser_wavelength_m':1064e-9,'detected_optical_power_W':.01,
      'quantum_efficiency':.8,'technical_phase_ASD_rad_per_sqrt_Hz':1e-8,
      'electronic_phase_ASD_rad_per_sqrt_Hz':3e-9}
    L=cfg['L_m'];wl=cfg['laser_wavelength_m']
    flux=cfg['quantum_efficiency']*cfg['detected_optical_power_W']/(h*c/wl)
    shot=np.sqrt(2/flux)
    total=np.sqrt(shot*shot+cfg['technical_phase_ASD_rad_per_sqrt_Hz']**2+cfg['electronic_phase_ASD_rad_per_sqrt_Hz']**2)
    f=np.linspace(1e3,12e6,2200)
    base=michelson_transfer(f,L)
    tilted=michelson_transfer(f,L,np.pi/3,0)
    tilted2=michelson_transfer(f,L,np.pi/3,np.pi/6)
    exactnormal=np.exp(2j*np.pi*f*L/c)*np.sinc(2*f*L/c)
    check=float(np.max(np.abs(base-exactnormal)))
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for val,label,color,style in [(base,'Нормальное падение',TEAL,'-'),(tilted,'Наклон 60°, азимут 0°',ORANGE,'--'),(tilted2,'Наклон 60°, азимут 30°',GRAY,':')]:
        axes[0].plot(f/1e6,np.abs(val),style,color=color,label=label)
    axes[0].set(xlabel='Частота, МГц',ylabel='Модуль геометрического отклика')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.15)
    response=4*np.pi*L/wl*np.abs(base)
    sens=total/response
    axes[1].semilogy(f/1e6,sens,color=TEAL)
    axes[1].axhline(total/(4*np.pi*L/wl),color=GRAY,ls=':',label='Длинноволновый предел')
    for n in [1,2,3]:axes[1].axvline(n*c/(2*L)/1e6,color=ORANGE,alpha=.3,lw=.8)
    axes[1].set(xlabel='Частота, МГц',ylabel='Условная ASD деформации, 1/√Гц',ylim=(1e-17,1e-14))
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.15)
    savefig(fig,'physics_interferometer_response')
    freqs=np.array([1e3,1e6,2e6,3e6,5e6,10e6])
    r=michelson_transfer(freqs,L)
    return {'status':'known TT plane-wave sector; supplied instrument noise budget, not a new quantum-gravity detection',
      'parameters':cfg,'Fourier_convention':'exp(-i omega t); detector time is return time',
      'effective_photon_flux_per_s':float(flux),'shot_phase_ASD_rad_per_sqrt_Hz':float(shot),
      'total_phase_ASD_rad_per_sqrt_Hz':float(total),'phase_response_low_frequency_rad_per_strain':float(4*np.pi*L/wl),
      'low_frequency_strain_ASD_per_sqrt_Hz':float(total/(4*np.pi*L/wl)),
      'first_normal_incidence_null_Hz':float(c/(2*L)),
      'normal_incidence_identity_max_absolute_error':check,
      'frequencies_Hz':freqs.tolist(),'transfer_moduli':np.abs(r).tolist(),
      'conditional_strain_ASD':(total/(4*np.pi*L/wl*np.abs(r))).tolist(),
      'free_mass_condition':'TT coordinates keep freely falling end mirrors fixed to first order; no extra mirror displacement is appended. Supported mirrors require a mechanical transfer function outside this free-mass band.'}


def main():
    result={'date':'2026-11-13','status':'conditional calculations and synthetic data; no measured new physics',
            'compact':compact_example(),'clock_gravimeter':clock_example(),
            'interferometer':interferometer_example(),'tidal_curvature':tidal_curvature_example()}
    (RESULTS/'physics_revision.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    short={k:{kk:vv for kk,vv in v.items() if not kk.startswith('curve_')} if isinstance(v,dict) else v for k,v in result.items()}
    print(json.dumps(short,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
