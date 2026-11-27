"""Bounded calibration and two-probe examples for the second book edition.

All data are synthetic. The global exclusions below use inverse Loewner
envelopes and simultaneous Gaussian marginal intervals, not a failed local
optimization. A local bounded fit supplies a witness of compatibility only.
Run: python revision2/code/math_revision.py
Search check only: python revision2/code/math_revision.py --search-validation
"""
from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np
from scipy.optimize import least_squares, lsq_linear, minimize_scalar
from scipy.stats import norm, chi2
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / 'revision2'
OUT = REV / 'results'
FIG = REV / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
N = 64
XYZ = np.array([(x,y,z) for x in range(4) for y in range(4) for z in range(4)])
EDGES = np.array([(i,j) for i in range(N) for j in range(i+1,N)
                  if np.abs(XYZ[i]-XYZ[j]).sum()==1])
PAIRS = np.array([(i,j) for i in range(N) for j in range(i+1,N)])
INC = np.zeros((N,len(EDGES)))
INC[EDGES[:,0],np.arange(len(EDGES))] = 1
INC[EDGES[:,1],np.arange(len(EDGES))] = -1
PAIR_INC = np.zeros((N,len(PAIRS)))
PAIR_INC[PAIRS[:,0],np.arange(len(PAIRS))] = 1
PAIR_INC[PAIRS[:,1],np.arange(len(PAIRS))] = -1
S = np.array([.1,.3,1.])
CORNERS = np.flatnonzero(np.all((XYZ==0)|(XYZ==3),axis=1))
FACE_PORTS = np.array([5,6,9,10,53,54,57,58])
EPS = .02
GMAX = 3.
ALPHA = .01

def laplacian(weights):
    return (INC*weights)@INC.T

def incidence(pair):
    b=np.zeros(N);b[pair[0]]=1;b[pair[1]]=-1
    return b

def contrasts(n):
    """Fixed unit norm readout vectors: diagonals and all pair differences."""
    out=[np.eye(n)[i] for i in range(n)]
    names=[f'node {i}' for i in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            out.append((np.eye(n)[i]-np.eye(n)[j])/np.sqrt(2))
            names.append(f'({i}-{j})/sqrt2')
    return np.array(out),names

class Probe:
    def __init__(self,name,weights,ports,s,eta=1.):
        self.name=name;self.weights=np.asarray(weights);self.ports=np.asarray(ports)
        self.s=np.asarray(s);self.eta=float(eta)
        self.L0=laplacian(self.weights)
        self.Q,self.qnames=contrasts(len(ports))
        self.A=np.zeros((len(self.Q),N));self.A[:,self.ports]=self.Q
        self.G0=np.array([np.linalg.inv(self.L0+x*np.eye(N)) for x in self.s])
        self.Glow=np.array([np.linalg.inv((1+EPS)*self.L0+x*np.eye(N)) for x in self.s])
        self.Ghigh=np.array([np.linalg.inv((1-EPS)*self.L0+x*np.eye(N)) for x in self.s])

    def response(self,delta=None,pair=None,g=0.):
        delta=np.zeros(len(EDGES)) if delta is None else np.asarray(delta)
        L=laplacian(self.weights*(1+delta))
        if pair is not None and g:
            b=incidence(pair);L=L+self.eta*g*np.outer(b,b)
        return np.array([np.linalg.inv(L+s*np.eye(N))[np.ix_(self.ports,self.ports)] for s in self.s])

    def baseline(self):
        return self.G0[:,self.ports][:,:,self.ports]

def make_probes():
    axis=np.argmax(np.abs(XYZ[EDGES[:,0]]-XYZ[EDGES[:,1]]),axis=1)
    p1=Probe('corner',np.ones(len(EDGES)),CORNERS,S)
    p2=Probe('anisotropic_face',np.array([2.,.7,1.4])[axis],FACE_PORTS,S,eta=1.6)
    dup=Probe('scaled_copy',2.5*np.ones(len(EDGES)),CORNERS,2.5*S,eta=2.5)
    return p1,p2,dup

def evaluate_bounds(probes,observations,sigmas,alpha=ALPHA,noise_multiplier=None):
    """Outer confidence set for one common pair with known positive couplings.

    Simultaneous coverage uses Bonferroni over fixed scalar contrasts, NOT
    independence of those contrasts. Independently acquired directed entries
    have sd sigma, so every unit-vector quadratic form has sd sigma. Bounds
    permit separate arbitrary conductance drift in each probe; shared drift is
    a narrower model, making these exclusions conservative for it as well.
    """
    count=sum(len(p.s)*len(p.Q) for p in probes)
    k=float(norm.ppf(1-alpha/(2*count))) if noise_multiplier is None else noise_multiplier
    lower=np.zeros(len(PAIRS));upper=np.full(len(PAIRS),GMAX)
    valid=np.ones(len(PAIRS),dtype=bool)
    cal_violations=[]
    diagnostics=[]
    for p,obs,sigma in zip(probes,observations,sigmas):
        for si,(lo,hi) in enumerate(zip(p.Glow,p.Ghigh)):
            zobs=np.einsum('ki,ij,kj->k',p.Q,obs[si],p.Q)
            zlo=np.einsum('ki,ij,kj->k',p.A,lo,p.A)
            zhi=np.einsum('ki,ij,kj->k',p.A,hi,p.A)
            ulo=p.A@lo@PAIR_INC;uhi=p.A@hi@PAIR_INC
            dlo=np.einsum('ik,ij,jk->k',PAIR_INC,lo,PAIR_INC)
            dhi=np.einsum('ik,ij,jk->k',PAIR_INC,hi,PAIR_INC)
            qmin=zobs-k*sigma;qmax=zobs+k*sigma
            cal_bad=(qmax<zlo)|(qmin>zhi)
            for qi in np.flatnonzero(cal_bad):
                cal_violations.append({'probe':p.name,'s':float(p.s[si]),'contrast':p.qnames[qi],
                                      'observed':float(zobs[qi]),'lower':float(zlo[qi]),'upper':float(zhi[qi]),
                                      'sigma':sigma,'signed_distance_sigma':float((zobs[qi]-np.clip(zobs[qi],zlo[qi],zhi[qi]))/sigma) if sigma else None})
            for qi in range(len(p.Q)):
                v2lo=ulo[qi]**2;v2hi=uhi[qi]**2
                atmaxlo=zlo[qi]-p.eta*GMAX*v2lo/(1+p.eta*GMAX*dlo)
                atmaxhi=zhi[qi]-p.eta*GMAX*v2hi/(1+p.eta*GMAX*dhi)
                valid&=(atmaxlo<=qmax[qi]+1e-13)&(zhi[qi]>=qmin[qi]-1e-13)
                dl=zlo[qi]-qmax[qi]
                if dl>0:
                    den=p.eta*(v2lo-dl*dlo)
                    req=np.divide(dl,den,out=np.full_like(den,np.inf),where=den>0)
                    lower=np.maximum(lower,req)
                dh=zhi[qi]-qmin[qi]
                cross=atmaxhi<qmin[qi]
                den=p.eta*(v2hi-dh*dhi)
                permitted=np.divide(dh,den,out=np.full_like(den,GMAX),where=(den>0)&cross)
                upper=np.minimum(upper,permitted)
            diagnostics.append({'probe':p.name,'s':float(p.s[si]),'observed':zobs.tolist(),
                                'cal_lower':zlo.tolist(),'cal_upper':zhi.tolist(),'sigma':sigma})
    valid&=(lower<=upper+1e-12)&(upper>=0)&(lower<=GMAX)
    candidates=[{'pair':list(map(int,PAIRS[i])),'g_interval':[float(max(0,lower[i])),float(min(GMAX,upper[i]))]}
                for i in np.flatnonzero(valid)]
    return {'fixed_scalar_contrasts':count,'alpha_family':alpha,'normal_multiplier':k,
            'calibration_excluded':bool(cal_violations),'calibration_violations':cal_violations,
            'remaining_pair_count':len(candidates),'candidates':candidates,'diagnostics':diagnostics,
            'interpretation':'Necessary outer conditions only; surviving pairs are not proofs of model existence.'}

def bounded_fit(probes,observations,sigmas,pair=None,start_g=.1,max_nfev=160):
    """Shared relative component drift; exact forward map and analytic Jacobian."""
    n=len(EDGES)+(pair is not None)
    pair_list=([pair]*len(probes) if pair is not None and isinstance(pair[0],(int,np.integer)) else pair)
    bs=[incidence(item) for item in pair_list] if pair is not None else [None]*len(probes)
    cache={}
    def calculate(x):
        if 'x' in cache and np.array_equal(cache['x'],x):return cache['r'],cache['J']
        residual=[];jacs=[]
        for p,obs,sigma,b in zip(probes,observations,sigmas,bs):
            L=laplacian(p.weights*(1+x[:len(EDGES)]))
            if b is not None:L=L+p.eta*x[-1]*np.outer(b,b)
            for si,s in enumerate(p.s):
                G=np.linalg.inv(L+s*np.eye(N));GP=G[p.ports]
                pred=GP[:,p.ports]
                residual.append(((pred-obs[si])/sigma).ravel())
                U=GP@INC
                jac=-np.einsum('ie,je,e->ije',U,U,p.weights).reshape(len(p.ports)**2,len(EDGES))
                if b is not None:
                    u=GP@b;jac=np.column_stack([jac,(-p.eta*np.outer(u,u)).ravel()])
                jacs.append(jac/sigma)
        r=np.concatenate(residual);J=np.vstack(jacs)
        cache.update({'x':x.copy(),'r':r,'J':J})
        return r,J
    lo=np.full(n,-EPS);hi=np.full(n,EPS);x0=np.zeros(n)
    if pair is not None:lo[-1]=0;hi[-1]=GMAX;x0[-1]=start_g
    result=least_squares(lambda x:calculate(x)[0],x0,jac=lambda x:calculate(x)[1],bounds=(lo,hi),
                         ftol=1e-10,xtol=1e-10,gtol=1e-8,max_nfev=max_nfev,method='trf')
    r=calculate(result.x)[0];Ndata=len(r);radius=float(chi2.ppf(.99,Ndata))
    return {'pair':list(pair) if pair is not None else None,'g':float(result.x[-1]) if pair is not None else 0.,
            'relative_weight_increments':result.x[:len(EDGES)].tolist(),
            'chi2':float(r@r),'data_count':Ndata,'chi2_per_entry':float(r@r/Ndata),
            'confidence_ball_chi2_99':radius,'found_feasible_point':bool(r@r<=radius),
            'solver_success':bool(result.success),'solver_message':result.message,'nfev':result.nfev,
            'scope':'Local constrained optimizer; success proves a numerical witness, failure does not prove global rejection.'}

def counterexample(p1):
    g=.03;b=incidence((5,58));base=p1.baseline()
    target=p1.response(pair=(5,58),g=g)-base
    J=np.stack([np.array([-np.outer(G[p1.ports,a]-G[p1.ports,bn],G[p1.ports,a]-G[p1.ports,bn])
                          for G in p1.G0]).ravel() for a,bn in EDGES],axis=1)
    fit=lsq_linear(J,target.ravel(),bounds=(-EPS,EPS),tol=1e-12,lsmr_tol=1e-13,max_iter=200)
    delta=fit.x;clean=p1.response(delta)
    noise=np.random.default_rng(271828).normal(0,1e-4,(4,8,8))[:3]
    observed=clean+noise
    return delta,clean,observed,target

def search_validation():
    spec=importlib.util.spec_from_file_location('old_study',ROOT/'book/study_v02.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    old=json.loads((ROOT/'book/search_validation.json').read_text())
    records=[]
    for rec in old:
        seed=rec['seed'];sigma=rec['sigma'];cal=rec['calibration'];rng=np.random.default_rng(seed)
        weights=np.exp(cal*rng.normal(size=len(m.EDGES))-.5*cal**2) if cal else np.ones(len(m.EDGES))
        L=m.laplacian(weights);b=incidence((5,58));L+=.3*np.outer(b,b)
        clean=np.array([(np.linalg.inv(L+s*np.eye(N))-base)[np.ix_(m.PORTS,m.PORTS)] for s,base in zip(m.S,m.G0)])
        observed=clean+rng.normal(0,sigma,clean.shape);fast=m.fit(observed,sigma)
        overlap=np.einsum('esi,sij,esj->es',m.U,observed[:3],m.U)
        best=(np.inf,None,None)
        for i,pair in enumerate(m.PAIRS):
            def objective(g):
                a=g/(1+g*m.D[i]);return float(np.sum(a*a*m.A[i]+2*a*overlap[i]))
            fit=minimize_scalar(objective,bounds=(0,3),method='bounded',options={'xatol':1e-12})
            value,g=min((objective(0),0.),(objective(3),3.),(float(fit.fun),float(fit.x)))
            if value<best[0]:best=(value,g,list(map(int,pair)))
        records.append({'seed':seed,'sigma':sigma,'calibration':cal,'fast_pair':fast['pair'],'full_pair':best[2],
                        'fast_g':fast['g'],'full_g':best[1],'g_difference':abs(fast['g']-best[1]),
                        'original_reported_g_difference':rec['g_difference'],
                        'pair_matches_original':fast['pair']==rec['fast_pair'] and best[2]==rec['full_pair'],
                        'strength_agreement_within_1e_7':abs(fast['g']-best[1])<1e-7})
    out={'records':records,'full_search':'2016 pairs, bounded scalar minimization on [0,3] plus both endpoints, xatol=1e-12',
         'provenance':'The original three seeds and exact data-generation order are reproduced. Last numerical optimizer digits are newly recorded, not asserted to be bitwise identical to a missing old script.',
         'scope':'Checks these datasets; not a universal proof of global scalar optimality.'}
    (OUT/'search_validation_reproduced.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    return out

def slim_bound(x):
    return {k:v for k,v in x.items() if k not in ['diagnostics','calibration_violations']}

def local_sensitivity():
    p0,_,_=make_probes();b=incidence((5,58))
    J=np.stack([np.array([-np.outer(G[p0.ports,a]-G[p0.ports,bn],G[p0.ports,a]-G[p0.ports,bn])
                         for G in p0.G0]).ravel() for a,bn in EDGES],axis=1)
    h=np.array([-np.outer(G[p0.ports]@b,G[p0.ports]@b) for G in p0.G0]).ravel()
    sv=np.linalg.svd(J,compute_uv=False);coef=np.linalg.lstsq(J,h,rcond=1e-10)[0]
    cutoff_sweep=[]
    for cutoff in [1e-6,1e-8,1e-10,1e-12]:
        trial=np.linalg.lstsq(J,h,rcond=cutoff)[0]
        cutoff_sweep.append({'relative_cutoff':cutoff,
                             'rank':int(np.sum(sv>sv[0]*cutoff)),
                             'projected_fraction':float(np.linalg.norm(h-J@trial)/np.linalg.norm(h)),
                             'maximum_abs_coefficient_per_unit_g':float(np.max(np.abs(trial)))})
    return {'shape':list(J.shape),'relative_svd_cutoff':1e-10,'rank':int(np.sum(sv>sv[0]*1e-10)),
            'projected_fraction':float(np.linalg.norm(h-J@coef)/np.linalg.norm(h)),
            'coefficient_norm_per_unit_g':float(np.linalg.norm(coef)),
            'maximum_abs_coefficient_per_unit_g':float(np.max(np.abs(coef))),
            'singular_values':sv.tolist(),
            'relative_singular_values':(sv/sv[0]).tolist(),
            'cutoff_sweep':cutoff_sweep,
            'qualification':'Unconstrained local derivative diagnostic only. Large coefficients invalidate any inference of finite mimicry by a small bounded drift.'}

def run():
    p0,proto,_=make_probes();rng=np.random.default_rng(730013)
    p1=Probe('left_face',np.ones(len(EDGES)),[0,3,5,6,9,10,12,15],S)
    p2=Probe('right_face',proto.weights,[48,51,53,54,57,58,60,63],S,eta=1.6)
    dup=Probe('scaled_copy',2.5*np.ones(len(EDGES)),p1.ports,2.5*S,eta=2.5)
    check_ports=np.array([0,5,6,15,48,57,58,63])
    c1=Probe('check_probe1',p1.weights,check_ports,[.3],eta=1.)
    c2=Probe('check_probe2',p2.weights,check_ports,[.3],eta=1.6)
    delta,clean,obs,target=counterexample(p0)
    counter_fit0=bounded_fit([p0],[obs],[1e-4])
    counter_fit1=bounded_fit([p0],[obs],[1e-4],pair=(5,58),start_g=.03)
    counter_bounds=evaluate_bounds([p0],[obs],[1e-4])
    true_delta=np.random.default_rng(730014).uniform(-.01,.01,len(EDGES))
    g=.3;pair=(5,58);sigma=1e-3
    mu1=p1.response(true_delta,pair,g);mu2=p2.response(true_delta,pair,g)
    mudup=dup.response(true_delta,pair,g)
    y1=mu1+rng.normal(0,sigma,mu1.shape);y2=mu2+rng.normal(0,sigma,mu2.shape)
    ydup=mudup+rng.normal(0,sigma/2.5,mudup.shape)
    one=evaluate_bounds([p1],[y1],[sigma]);second=evaluate_bounds([p2],[y2],[sigma]);two=evaluate_bounds([p1,p2],[y1,y2],[sigma,sigma])
    copy=evaluate_bounds([p1,dup],[y1,ydup],[sigma,sigma/2.5])
    noiseless_one=evaluate_bounds([p1],[mu1],[0.],noise_multiplier=0.)
    noiseless_dup=evaluate_bounds([p1,dup],[mu1,mudup],[0.,0.],noise_multiplier=0.)
    fit0=bounded_fit([p1,p2],[y1,y2],[sigma,sigma])
    fit1=bounded_fit([p1,p2],[y1,y2],[sigma,sigma],pair=pair,start_g=.25)
    # A physically distinct second support, kept outside any common one-pair
    # hypothesis. Shared relative weight drift is unchanged.
    different_pair=(6,57)
    mu_different=p2.response(true_delta,different_pair,g)
    ydifferent=mu_different+rng.normal(0,sigma,mu_different.shape)
    inconsistent=evaluate_bounds([p1,p2],[y1,ydifferent],[sigma,sigma])
    second_separate=evaluate_bounds([p2],[ydifferent],[sigma])
    hybrid_fit=None
    if len(inconsistent['candidates'])==1:
        hybrid_fit=bounded_fit([p1,p2],[y1,ydifferent],[sigma,sigma],pair=inconsistent['candidates'][0]['pair'],start_g=.3)
    separate_fit=bounded_fit([p1,p2],[y1,ydifferent],[sigma,sigma],pair=[pair,different_pair],start_g=.3)
    validation_rng=np.random.default_rng(934101)
    yc1=c1.response(true_delta,pair,g)+validation_rng.normal(0,sigma,(1,8,8))
    yc2=c2.response(true_delta,pair,g)+validation_rng.normal(0,sigma,(1,8,8))
    ycd=c2.response(true_delta,different_pair,g)+validation_rng.normal(0,sigma,(1,8,8))
    common_confirm=evaluate_bounds([p1,p2,c1,c2],[y1,y2,yc1,yc2],[sigma]*4)
    different_confirm=evaluate_bounds([p1,p2,c1,c2],[y1,ydifferent,yc1,ycd],[sigma]*4)
    common_confirm_fit=bounded_fit([p1,p2,c1,c2],[y1,y2,yc1,yc2],[sigma]*4,pair=pair,start_g=.3)
    separate_confirm_fit=bounded_fit([p1,p2,c1,c2],[y1,ydifferent,yc1,ycd],[sigma]*4,pair=[pair,different_pair,pair,different_pair],start_g=.3)
    # Common electronic leakage is deliberately shaped like probe 1's edge
    # response and copied in channel coordinates to probe 2. A reference run
    # with the physical propagation input disabled records that same leakage.
    artifact=p1.response(pair=pair,g=g)-p1.baseline()
    ya1=p1.baseline()+artifact+rng.normal(0,sigma,artifact.shape)
    ya2=p2.baseline()+artifact+rng.normal(0,sigma,artifact.shape)
    artifact_joint=evaluate_bounds([p1,p2],[ya1,ya2],[sigma,sigma])
    blank=artifact+rng.normal(0,sigma,artifact.shape)
    blank_norm=float(np.sum(blank**2)/sigma**2)
    corrected1=ya1-blank;corrected2=ya2-blank
    corrected=evaluate_bounds([p1,p2],[corrected1,corrected2],[np.sqrt(2)*sigma,np.sqrt(2)*sigma])
    # Fresh independent noise records, same prespecified fixed design; never
    # estimate a universal error rate from this small conditional ensemble.
    trials=[]
    for kind in ['bounded_drift','common_edge','different_edges','copied_leakage']:
        counts={'calibration_excluded':0,'no_common_pair':0,'true_pair_retained':0,
                'after_control_calibration_excluded':0,'after_control_no_common_pair':0,'after_control_true_pair_retained':0}
        per=[]
        for k in range(24):
            rr=np.random.default_rng(3600000+1000*['bounded_drift','common_edge','different_edges','copied_leakage'].index(kind)+k)
            dw=rr.uniform(-EPS,EPS,len(EDGES))
            if kind=='bounded_drift':m1=p1.response(dw);m2=p2.response(dw)
            elif kind=='common_edge':m1=p1.response(dw,pair,g);m2=p2.response(dw,pair,g)
            elif kind=='different_edges':m1=p1.response(dw,pair,g);m2=p2.response(dw,different_pair,g)
            else:m1=p1.response(dw)+artifact;m2=p2.response(dw)+artifact
            yy1=m1+rr.normal(0,sigma,m1.shape);yy2=m2+rr.normal(0,sigma,m2.shape)
            bb=evaluate_bounds([p1,p2],[yy1,yy2],[sigma,sigma])
            if kind=='bounded_drift':mc1=c1.response(dw);mc2=c2.response(dw)
            elif kind=='common_edge':mc1=c1.response(dw,pair,g);mc2=c2.response(dw,pair,g)
            elif kind=='different_edges':mc1=c1.response(dw,pair,g);mc2=c2.response(dw,different_pair,g)
            else:mc1=c1.response(dw)+artifact[1:2];mc2=c2.response(dw)+artifact[1:2]
            yyc1=mc1+rr.normal(0,sigma,mc1.shape);yyc2=mc2+rr.normal(0,sigma,mc2.shape)
            bc=evaluate_bounds([p1,p2,c1,c2],[yy1,yy2,yyc1,yyc2],[sigma]*4)
            kept=any(c['pair']==list(pair) for c in bb['candidates'])
            counts['calibration_excluded']+=bb['calibration_excluded']
            counts['no_common_pair']+=(bb['remaining_pair_count']==0)
            counts['true_pair_retained']+=kept
            counts['after_control_calibration_excluded']+=bc['calibration_excluded']
            counts['after_control_no_common_pair']+=(bc['remaining_pair_count']==0)
            counts['after_control_true_pair_retained']+=any(c['pair']==list(pair) for c in bc['candidates'])
            per.append({'seed':3600000+1000*['bounded_drift','common_edge','different_edges','copied_leakage'].index(kind)+k,
                        'calibration_excluded':bb['calibration_excluded'],'remaining_pairs':bb['remaining_pair_count'],
                        'true_pair_retained':kept if kind=='common_edge' else None,
                        'after_control_calibration_excluded':bc['calibration_excluded'],
                        'after_control_remaining_pairs':bc['remaining_pair_count'],
                        'after_control_true_pair_retained':any(c['pair']==list(pair) for c in bc['candidates']) if kind=='common_edge' else None})
        if kind!='common_edge':
            counts['true_pair_retained']=None
            counts['after_control_true_pair_retained']=None
        trials.append({'scenario':kind,'n':24,**counts,'trials':per})
    # Prespecified holdout s=.3 with face ports, independently acquired record.
    hold_s=np.array([.1,.3,1.,3.])
    signal_norm={}
    for p in [p0,p1,p2]:
        pp=Probe(p.name,p.weights,p.ports,hold_s,p.eta)
        signal=pp.response(pair=pair,g=g)-pp.baseline()
        signal_norm[p.name]={'s':hold_s.tolist(),'rho_at_sigma_1e_4':(np.linalg.norm(signal,axis=(1,2))/1e-4).tolist(),
                             'lambda_at_sigma_1e_4':(np.sum(signal**2,axis=(1,2))/1e-8).tolist()}
    # Generalization control: fixed candidate and strength from probe 1 only,
    # evaluated against a fresh independent acquisition of probe 2 at s=.3.
    ftrain=bounded_fit([p1],[y1],[sigma],pair=pair,start_g=.2)
    dhat=np.array(ftrain['relative_weight_increments']);ghat=ftrain['g']
    true_hold=p2.response(true_delta,pair,g)[1]
    pred_hold=p2.response(dhat,pair,ghat)[1]
    yhold=true_hold+np.random.default_rng(930001).normal(0,sigma,(8,8))
    null_hold=p2.baseline()[1]
    hold={'source':'probe1-only bounded fit predicts separately acquired probe2 at s=.3',
          'predeclared_seed':930001,'sigma':sigma,'fitted_g_from_probe1':ghat,
          'prediction_chi2_per_entry':float(np.sum((yhold-pred_hold)**2)/(64*sigma**2)),
          'nominal_no_edge_chi2_per_entry':float(np.sum((yhold-null_hold)**2)/(64*sigma**2)),
          'clean_prediction_vs_true_lambda':float(np.sum((pred_hold-true_hold)**2)/sigma**2),
          'true_signal_vs_nominal_null_lambda':float(np.sum((true_hold-null_hold)**2)/sigma**2),
          'qualification':'This prediction can fail when the probe1 drift estimate is poorly determined; the separate global bounds remain valid.'}
    result={'status':'Synthetic bounded-model demonstrations, second edition',
            'design':{'nodes':N,'ordinary_edges':len(EDGES),'candidate_pairs':len(PAIRS),'eps':EPS,'g_bounds':[0,GMAX],
                      's':S.tolist(),'alpha_family':ALPHA,'probes':[{'name':p.name,'ports':p.ports.tolist(),'eta':p.eta,
                       'weights':p.weights.tolist(),'s':p.s.tolist()} for p in [p0,p1,p2,dup,c1,c2]],
                      'noise_model':'Independent Gaussian errors for separately acquired directed entries, scales and probe records. Contrasts within one matrix are correlated; Bonferroni needs only their marginal variances.'},
            'counterexample':{'relative_weights':delta.tolist(),'observations':obs.tolist(),'calibration_fit':counter_fit0,
                              'edge_fit':counter_fit1,'outer_bounds':slim_bound(counter_bounds)},
            'common_edge':{'true_pair':list(pair),'true_g':g,'relative_weights':true_delta.tolist(),'sigma':sigma,
                           'one_probe':slim_bound(one),'second_probe':slim_bound(second),'two_probes':slim_bound(two),'scaled_copy':slim_bound(copy),
                           'noiseless_one':slim_bound(noiseless_one),'noiseless_scaled':slim_bound(noiseless_dup),
                           'scaling_error':float(np.linalg.norm(2.5*mudup-mu1)/np.linalg.norm(mu1)),
                           'calibration_fit':fit0,'edge_fit':fit1,
                           'with_control_ports':slim_bound(common_confirm),'with_control_fit':common_confirm_fit},
            'different_edges':{'true_pairs':[list(pair),list(different_pair)],'joint':slim_bound(inconsistent),
                               'second_separate':slim_bound(second_separate),'hybrid_common_fit':hybrid_fit,'separate_support_fit':separate_fit,
                               'with_control_ports':slim_bound(different_confirm),'with_control_separate_fit':separate_confirm_fit},
            'common_leakage':{'joint':slim_bound(artifact_joint),'blank_chi2':blank_norm,'blank_data_count':blank.size,
                              'blank_99_threshold':float(chi2.ppf(.99,blank.size)),'after_reference_subtraction':slim_bound(corrected),
                              'qualification':'The reference record is shared by the two subtractions; their errors are correlated. Marginal sd sqrt2*sigma and union coverage are still valid.'},
            'fresh_noise_series':trials,'holdout_design':signal_norm,'separate_holdout':hold,
            'local_sensitivity':local_sensitivity(),
            'search_validation':search_validation(),
            'limits':['Conductance box is an assumption requiring independent calibration in hardware',
                      'One extra nonnegative edge with known relative coupling between probes',
                      'Known node correspondence, masses, topology and readout maps',
                      'Envelope survivors need not be jointly realizable by one conductance vector',
                      'Bounded optimizer is local; it supplies witnesses but not global rejection',
                      'No experimental measurement of physical space; no novelty priority claim']}
    (OUT/'math_revision.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    make_figures(result,p0,p1,p2,delta,target,one,two,copy,y1,y2)
    return result

def make_figures(result,p0,p1,p2,delta,target,one,two,copy,y1,y2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    drift=p0.response(delta)-p0.baseline()
    fig=plt.figure(figsize=(10.8,3.6),layout='constrained')
    grid=fig.add_gridspec(1,4,width_ratios=[1,1,1,.045])
    axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
    vmax=max(np.max(np.abs(drift[0])),np.max(np.abs(target[0])))
    for ax,data,title in zip(axes,[target[0],drift[0],(drift[0]-target[0])],['Добавленная связь g = 0,03','Только обычный дрейф ±2%','Разность двух откликов']):
        im=ax.imshow(data,cmap='RdBu_r',vmin=-vmax,vmax=vmax);ax.set_title(title,fontsize=10);ax.set_xlabel('Порт источника');ax.set_ylabel('Порт считывания')
    fig.colorbar(im,cax=fig.add_subplot(grid[0,3]),label='ΔG, с')
    fig.savefig(FIG/'math_counterexample.png',dpi=220,bbox_inches='tight');fig.savefig(FIG/'math_counterexample.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3.6));x=np.arange(4);width=.35
    for off,key,label,color in [(-width/2,'corner','Угловые порты','#306977'),(width/2,'right_face','Второй зонд и новые порты','#b16c3a')]:
        vals=result['holdout_design'][key]['rho_at_sigma_1e_4'];ax.bar(x+off,vals,width,label=label,color=color)
    ax.set_xticks(x,['0,1','0,3','1','3']);ax.set_xlabel('s, с⁻¹');ax.set_ylabel('Норма сигнала / σ');ax.set_yscale('log');ax.legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.22),ncol=1);save(fig,'math_holdout_power')
    fig,ax=plt.subplots(figsize=(7.2,3.6));names=['Первый зонд','Масштабированная копия','Иной зонд и порты'];values=[one['remaining_pair_count'],copy['remaining_pair_count'],two['remaining_pair_count']]
    ax.bar(range(3),values,color=['#306977','#70888d','#b16c3a']);ax.set_xticks(range(3),names,fontsize=9);ax.set_ylabel('Пар, не исключённых оболочками');
    for i,v in enumerate(values):ax.text(i,v+max(values)*.02,str(v),ha='center')
    ax.set_ylim(0,max(values)*1.18+1);save(fig,'math_two_probe_candidates')
    # Chosen contrast is part of the fixed family, not selected for discovery.
    pp=p2;a=np.eye(N)[58];b=incidence((5,58));si=1
    x=np.linspace(0,.65,300);lines=[]
    for G in [pp.Glow[si],pp.Ghigh[si]]:
        z0=a@G@a;u=a@G@b;d=b@G@b;lines.append(z0-pp.eta*x*u*u/(1+pp.eta*x*d))
    q=np.eye(8)[5];observed=q@y2[si]@q;k=two['normal_multiplier'];sigma=result['common_edge']['sigma']
    fig,ax=plt.subplots(figsize=(7.2,3.6));ax.fill_between(x,lines[0],lines[1],color='#306977',alpha=.25,label='Все веса в пределах ±2%');ax.plot(x,lines[0],color='#306977',lw=1);ax.plot(x,lines[1],color='#306977',lw=1)
    ax.axhspan(observed-k*sigma,observed+k*sigma,color='#b16c3a',alpha=.2,label='Одновременный интервал наблюдения');ax.axhline(observed,color='#b16c3a',lw=1);ax.axvline(.3,color='black',ls=':',lw=1,label='Внесённое g = 0,3')
    ax.set_xlabel('Общая интенсивность g, с⁻¹');ax.set_ylabel('Контраст aᵀG₂a, с');ax.legend(frameon=False,fontsize=9);save(fig,'math_bounded_envelope')
    fig,ax=plt.subplots(figsize=(7.2,3.6));groups=result['fresh_noise_series'];xx=np.arange(len(groups));bw=.35
    ax.bar(xx-bw/2,[g['after_control_calibration_excluded'] for g in groups],bw,label='Исключена чистая калибровка',color='#306977');ax.bar(xx+bw/2,[g['after_control_no_common_pair'] for g in groups],bw,label='Нет общей допустимой пары',color='#b16c3a')
    ax.set_xticks(xx,['Дрейф','Общая связь','Разные связи','Общая утечка']);ax.set_ylabel('Число случаев из 24');ax.set_ylim(0,28);ax.legend(frameon=False,fontsize=8);save(fig,'math_controls')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--search-validation',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=1):
        output=search_validation() if args.search_validation else run()
    if args.search_validation:print(json.dumps(output,ensure_ascii=False,indent=2))
    else:
        small={'counter_fit':{k:v for k,v in output['counterexample']['calibration_fit'].items() if k!='relative_weight_increments'},
               'counts':{key:output['common_edge'][key]['remaining_pair_count'] for key in ['one_probe','two_probes','scaled_copy','noiseless_one','noiseless_scaled']},
               'common_fit':{k:v for k,v in output['common_edge']['edge_fit'].items() if k!='relative_weight_increments'},
               'different_count':output['different_edges']['joint']['remaining_pair_count'],
               'leakage_count':output['common_leakage']['joint']['remaining_pair_count'],
               'series':[{k:v for k,v in r.items() if k!='trials'} for r in output['fresh_noise_series']],
               'holdout':output['separate_holdout']}
        print(json.dumps(small,ensure_ascii=False,indent=2))
