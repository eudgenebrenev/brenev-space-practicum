"""Reproducible four-node RC practicum. All supplied voltage records are SYNTHETIC.

Run from the extracted materials root:
    python revision2/code/rc_practicum.py all
    python revision2/code/rc_practicum.py analyse

`generate` creates CSV observations, independently specified calibration and a
separate truth key. `analyse` reads only calibration.json and sample CSVs; it
does not read truth_key.json. `all` performs both steps, then makes a separate
truth comparison for the explanatory figures. The finite-record estimator
uses the known connected-network zero mode. Gaussian sample noise, initial
voltage matrix and homogeneous loading are explicit assumptions of this
synthetic exercise, not specifications of tested hardware.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.linalg import eigh, cholesky, solve_triangular
from scipy.optimize import least_squares
from scipy.stats import chi2
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rc"
OUT = ROOT / "results"
FIG = ROOT / "figures"
N = 4
EDGES = [(0, 1), (1, 2), (2, 3)]
CANDIDATES = [(0, 2), (0, 3), (1, 3)]
P0 = np.ones((N, N)) / N


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def laplacian(weights, rate=100., pair=None, edge_fraction=0.):
    L = np.zeros((N, N))
    for (i, j), w in zip(EDGES, weights):
        b = np.eye(N)[i] - np.eye(N)[j]
        L += rate * w * np.outer(b, b)
    if pair is not None:
        b = np.eye(N)[pair[0]] - np.eye(N)[pair[1]]
        L += rate * edge_fraction * np.outer(b, b)
    return L


def evolution(L, times):
    vals, vecs = eigh(L)
    return np.einsum("ik,tk,jk->tij", vecs, np.exp(-times[:, None]*vals), vecs)


def simpson_weights(times):
    steps = np.diff(times)
    if len(times) % 2 != 1 or not np.allclose(steps, steps[0], atol=1e-13, rtol=0):
        raise ValueError("Uniform time grid with an even number of intervals required")
    q = np.ones(len(times)); q[1:-1:2] = 4.; q[2:-1:2] = 2.
    return q * steps[0] / 3.


def generate(data_dir=DATA):
    data_dir.mkdir(parents=True, exist_ok=True)
    W = np.diag([1.0002, .9998, 1.0001, .9999])
    for i, j in EDGES:
        W[i, j] = W[j, i] = .0015
    cal = {
        "status": "SYNTHETIC input records; exact calibration within this exercise",
        "nodes": 4, "ordinary_edges": EDGES, "ordinary_resistance_ohm": 100000.,
        "capacitance_F": [1e-7]*4, "nominal_rate_per_s": 100.,
        "input_resistance_ohm": 1e9, "homogeneous_loading_rate_per_s": .01,
        "sample_interval_s": .00005, "record_duration_s": .25,
        "initial_voltage_matrix_V": W.tolist(),
        "initial_matrix_convention": "rows=detector node, columns=prepared source state",
        "sample_noise_std_V": .0002,
        "noise_model": "independent Gaussian per raw sample, detector and preparation; after unmixing and Laplace integration the full covariance is propagated",
        "laplace_s_per_s": [20., 60., 200.],
        "ordinary_weight_bounds": [.98, 1.02],
        "candidate_added_edges": CANDIDATES, "added_edge_fraction_bounds": [0., .6],
        "primary_contrast": [float(1/np.sqrt(2)), 0., 0., float(-1/np.sqrt(2))],
        "primary_s_index": 0, "primary_gaussian_margin_sigma": 5.,
        "acceptance_rule": "Use only the preregistered scalar contrast at s=20 for global rejection of the bounded local network class; fit residuals are diagnostic witnesses, not global exclusion certificates.",
        "source_files": ["sample_A", "sample_B", "sample_C"],
        "all_nominal_values_are": "calculation inputs, not measured hardware performance",
    }
    json_write(data_dir / "calibration.json", cal)
    cases = [
        {"meaning": "nominal local network", "weights": [1., 1., 1.], "pair": None, "edge_fraction": 0.},
        {"meaning": "bounded local drift", "weights": [1.015, .985, 1.01], "pair": None, "edge_fraction": 0.},
        {"meaning": "bounded drift plus added edge", "weights": [1.01, .99, 1.005], "pair": [0, 3], "edge_fraction": .2},
    ]
    order = np.random.default_rng(94013).permutation(3)
    times = np.arange(round(cal["record_duration_s"]/cal["sample_interval_s"])+1)*cal["sample_interval_s"]
    truth = {"status": "Generator truth, not read by analyse", "seed_for_labels": 94013, "samples": {}}
    for k, label in enumerate(cal["source_files"]):
        case = cases[int(order[k])]
        L = laplacian(case["weights"], pair=case["pair"], edge_fraction=case["edge_fraction"])
        clean = np.einsum("tij,jk->tik", evolution(L + .01*np.eye(N), times), W)
        seed = 94101 + k
        noisy = clean + np.random.default_rng(seed).normal(0, cal["sample_noise_std_V"], clean.shape)
        for source in range(N):
            array = np.column_stack([times, noisy[:, :, source]])
            np.savetxt(data_dir / f"{label}_source_{source}.csv", array, delimiter=",",
                       header="time_s,v0_V,v1_V,v2_V,v3_V", comments="", fmt="%.12g")
        truth["samples"][label] = {**case, "noise_seed": seed}
    json_write(data_dir / "truth_key.json", truth)
    return cal


def read_sample(label, data_dir):
    records = []
    times = None
    for source in range(N):
        arr = np.loadtxt(data_dir / f"{label}_source_{source}.csv", delimiter=",", skiprows=1)
        if times is None:
            times = arr[:, 0]
        elif not np.array_equal(times, arr[:, 0]):
            raise ValueError("Preparation time grids do not match")
        records.append(arr[:, 1:])
    return times, np.stack(records, axis=-1)


def estimate_response(times, V, cal):
    """The observed V has indices (sample, detector, prepared source)."""
    W = np.asarray(cal["initial_voltage_matrix_V"])
    M = np.linalg.inv(W)
    sgrid = np.asarray(cal["laplace_s_per_s"])
    gamma = cal["homogeneous_loading_rate_per_s"]
    q = simpson_weights(times)
    E = np.einsum("tij,jk->tik", V, M)*np.exp(gamma*times[:, None, None])
    weights = q[None, :]*np.exp(-sgrid[:, None]*times[None, :])
    G = P0[None, :, :]/sgrid[:, None, None] + np.einsum("st,tij->sij", weights, E-P0)
    raw_weights = q[None, :]*np.exp(-(sgrid[:, None]-gamma)*times[None, :])
    Ctime = raw_weights @ raw_weights.T
    Csource = M.T @ M
    C = cal["sample_noise_std_V"]**2 * np.kron(Ctime, np.kron(np.eye(N), Csource))
    corr = Ctime/np.sqrt(np.diag(Ctime)[:, None]*np.diag(Ctime)[None, :])
    return G, C, corr, E


def analyse(data_dir=DATA):
    """Analyse only preregistered calibration and observed CSVs; never truth key."""
    cal = json.loads((data_dir / "calibration.json").read_text())
    sgrid = np.asarray(cal["laplace_s_per_s"])
    rate = cal["nominal_rate_per_s"]
    low, high = cal["ordinary_weight_bounds"]
    L0 = laplacian(np.ones(3), rate)
    vals0 = np.linalg.eigvalsh(L0)
    a = np.asarray(cal["primary_contrast"])
    lower = np.array([a @ np.linalg.solve(s*np.eye(N)+high*L0, a) for s in sgrid])
    upper = np.array([a @ np.linalg.solve(s*np.eye(N)+low*L0, a) for s in sgrid])
    output = {
        "status": "Analysis of SYNTHETIC voltage files; calibration assumptions explicit",
        "data_directory": "revision2/data/rc", "calibration": cal,
        "nominal_eigenvalues_per_s": vals0.tolist(),
        "zero_mode": "known connected ungrounded graph; homogeneous measurement loading corrected independently",
        "initial_matrix_condition_number": float(np.linalg.cond(np.asarray(cal["initial_voltage_matrix_V"]))),
        "normalization": "G(s) has seconds; H_VI(s)=G(s)/C0 has ohms",
        "covariance_order": "flatten (Laplace index, detector node, source preparation)",
        "contrast_lower_s": lower.tolist(), "contrast_upper_s": upper.tolist(),
        "samples": {},
    }
    for label in cal["source_files"]:
        times, V = read_sample(label, data_dir)
        G, C, corr, E = estimate_response(times, V, cal)
        chol = cholesky(C, lower=True)
        def model(parameters, pair):
            L = laplacian(parameters[:3], rate, pair, parameters[3] if pair is not None else 0.)
            return np.array([np.linalg.inv(s*np.eye(N)+L) for s in sgrid])
        def whitened(parameters, pair):
            return solve_triangular(chol, (model(parameters, pair)-G).ravel(), lower=True)
        local = least_squares(whitened, np.ones(3), args=(None,), bounds=([low]*3,[high]*3),
                              xtol=1e-12, ftol=1e-12, gtol=1e-10, max_nfev=1000)
        candidates=[]
        for pair in CANDIDATES:
            opt = least_squares(whitened, [1.,1.,1.,.1], args=(pair,),
                    bounds=([low]*3+[0.],[high]*3+[.6]),
                    xtol=1e-12, ftol=1e-12, gtol=1e-10, max_nfev=1000)
            candidates.append({"pair":list(pair), "ordinary_weights":opt.x[:3].tolist(),
                               "edge_fraction":float(opt.x[3]), "edge_rate_per_s":float(rate*opt.x[3]),
                               "chi2":float(2*opt.cost), "chi2_per_observation":float(2*opt.cost/48),
                               "solver_success":bool(opt.success), "nfev":int(opt.nfev)})
        candidates.sort(key=lambda item:item["chi2"])
        z = np.einsum("i,sij,j->s", a, G, a)
        qcontrast=np.outer(a,a).ravel()
        contrast_cov=np.array([[qcontrast @ C[i*16:(i+1)*16,j*16:(j+1)*16] @ qcontrast
                                for j in range(3)] for i in range(3)])
        sigma_z=np.sqrt(np.diag(contrast_cov))
        T=times[-1]; dt=times[1]-times[0]
        tail=np.exp(-(sgrid+low*vals0[1])*T)/(sgrid+low*vals0[1])
        # Composite Simpson absolute error bound for the scalar exponential mixture.
        # Under Hcal: eigenvalues <= high*lambda_max(L0), coefficient sum ||a||²=1.
        quad=(T/180)*dt**4*(sgrid+high*vals0[-1])**4
        index=cal["primary_s_index"]
        margin=cal["primary_gaussian_margin_sigma"]*sigma_z[index]+tail[index]+quad[index]
        reject=bool(z[index]<lower[index]-margin or z[index]>upper[index]+margin)
        # The scalar envelope is necessary, not sufficient for a joint fit.
        # A concrete bounded parameter vector can witness compatibility with
        # the joint Gaussian noise ellipsoid. Retain 48 dimensions: do not
        # subtract fitted parameters or apply a search-discovery interpretation.
        # The deterministic integration bias is included by the triangle bound.
        smallest_cov_eigenvalue=float(np.linalg.eigvalsh(C)[0])
        joint_noise_radius=float(np.sqrt(chi2.ppf(.99,G.size)))
        local_bias_radius=float(np.sqrt((N-1)*np.sum((tail+quad)**2)/smallest_cov_eigenvalue))
        local_joint_radius=joint_noise_radius+local_bias_radius
        local_witness=bool(np.linalg.norm(local.fun)<=local_joint_radius)
        added_quad=(T/180)*dt**4*(sgrid+high*vals0[-1]+2*rate*.6)**4
        added_bias_radius=float(np.sqrt((N-1)*np.sum((tail+added_quad)**2)/smallest_cov_eigenvalue))
        added_joint_radius=joint_noise_radius+added_bias_radius
        for candidate in candidates:
            candidate["joint_99pct_ellipsoid_radius_with_bias"]=added_joint_radius
            candidate["compatibility_witness"]=bool(np.sqrt(candidate["chi2"])<=added_joint_radius)
        if reject:
            verdict=("bounded local-network model rejected by preregistered scalar envelope; a bounded added-edge fit provides a compatibility witness"
                     if candidates[0]["compatibility_witness"] else
                     "bounded local-network model rejected; no compatible added-edge witness found by this local search")
        else:
            verdict=("a bounded local-network fit provides a compatibility witness; no added edge required, weak edges not excluded"
                     if local_witness else
                     "scalar envelope does not exclude the local class, but no joint compatibility witness found; unresolved")
        item={"G_estimated_s":G.tolist(), "covariance_s2":C.tolist(),
              "laplace_noise_correlation":corr.tolist(), "contrast_s":z.tolist(),
              "contrast_std_s":sigma_z.tolist(), "contrast_covariance_s2":contrast_cov.tolist(),
              "finite_tail_bound_s":tail.tolist(), "simpson_bound_under_Hcal_s":quad.tolist(),
              "primary_total_margin_s":float(margin), "Hcal_rejected_by_global_envelope":reject,
              "local_fit":{"ordinary_weights":local.x.tolist(),"chi2":float(2*local.cost),
                           "chi2_per_observation":float(2*local.cost/48),"solver_success":bool(local.success),
                           "joint_99pct_ellipsoid_radius_with_bias":local_joint_radius,
                           "deterministic_bias_whitened_radius_bound":local_bias_radius,
                           "compatibility_witness":local_witness},
              "added_edge_fits":candidates, "verdict":verdict,
              "raw_data_count":int(V.size), "samples_per_preparation":len(times),
              "reciprocity_residual_F_s":float(np.linalg.norm(G-G.transpose(0,2,1)))}
        output["samples"][label]=item
    json_write(OUT / "rc_analysis.json", output)
    return output


def evaluate_and_plot(data_dir=DATA):
    """After the frozen analysis, reveal generator truth for teaching figures."""
    result=json.loads((OUT/"rc_analysis.json").read_text())
    truth=json.loads((data_dir/"truth_key.json").read_text())
    cal=result["calibration"]
    sgrid=np.asarray(cal["laplace_s_per_s"])
    a=np.asarray(cal["primary_contrast"])
    evaluation={"status":"Post-analysis truth comparison, SYNTHETIC data", "samples":{}}
    for label, truth_item in truth["samples"].items():
        L=laplacian(truth_item["weights"], pair=truth_item["pair"], edge_fraction=truth_item["edge_fraction"])
        exact=np.array([np.linalg.inv(s*np.eye(N)+L) for s in sgrid])
        estimate=np.asarray(result["samples"][label]["G_estimated_s"])
        times,V=read_sample(label,data_dir)
        clean=np.einsum("tij,jk->tik", evolution(L+.01*np.eye(N),times),np.asarray(cal["initial_voltage_matrix_V"]))
        noiseless,C,_,_=estimate_response(times,clean,cal)
        chol=cholesky(C,lower=True)
        nominal=laplacian(np.ones(3))
        E=evolution(L,times)
        P=np.trace(E,axis1=1,axis2=2)/N
        partial=(E[:,0,0]+E[:,3,3])/2
        evaluation["samples"][label]={"truth":truth_item,
            "relative_G_error_F":float(np.linalg.norm(estimate-exact)/np.linalg.norm(exact)),
            "maximum_G_error_s":float(np.max(np.abs(estimate-exact))),
            "whitened_true_residual_chi2":float(np.sum(solve_triangular(chol,(estimate-exact).ravel(),lower=True)**2)),
            "noiseless_quadrature_and_tail_error_F_s":float(np.linalg.norm(noiseless-exact)),
            "contrast_exact_s":np.einsum("i,sij,j->s",a,exact,a).tolist(),
            "partial_vs_full_heat_trace_max_absolute":float(np.max(np.abs(partial-P))),
            "added_edge_resistance_ohm":None if truth_item["edge_fraction"]==0 else float(1/(1e-7*100*truth_item["edge_fraction"]))}
    # A single conditional torsion budget, independent of any claim of new force.
    kB=1.380649e-23; temp=300.; inertia=1e-6; f0=.01; Q=1000.; f=.005
    omega0=2*np.pi*f0; omega=2*np.pi*f
    kappa=inertia*omega0**2; damping=inertia*omega0/Q
    chi=1/(kappa-inertia*omega**2+1j*damping*omega)
    angle_asd=1e-8; tau_th=np.sqrt(4*kB*temp*damping); tau_ro=angle_asd/abs(chi)
    duration=100000.; amp_std=np.sqrt((tau_th**2+tau_ro**2)/duration)
    evaluation["conditional_torsion_budget"]={"status":"assumed viscous equilibrium model, no measured apparatus",
        "temperature_K":temp,"inertia_kg_m2":inertia,"f0_Hz":f0,"Q":Q,"drive_Hz":f,
        "kappa_N_m_per_rad":kappa,"damping_N_m_s_per_rad":damping,
        "susceptibility_abs_rad_per_N_m":float(abs(chi)),"angle_readout_asd_rad_per_sqrt_Hz":angle_asd,
        "thermal_torque_asd_N_m_per_sqrt_Hz":float(tau_th),"readout_torque_asd_N_m_per_sqrt_Hz":float(tau_ro),
        "duration_s":duration,"known_phase_torque_amplitude_std_N_m":float(amp_std),
        "excluded_from_budget":["gap error","source mass uncertainty","seismic coupling","electrostatic background","structural rather than viscous damping"]}
    json_write(OUT/"rc_validation.json",evaluation)
    make_figures(result,truth,evaluation,data_dir)
    return evaluation


def savefig(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/f"{name}.png",dpi=220,bbox_inches="tight")
    fig.savefig(FIG/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)


def make_figures(result,truth,evaluation,data_dir):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    blue="#346575";orange="#b96632";gray="#64737b"
    # Schematic: actual four-node topology and the four independent preparations.
    fig,ax=plt.subplots(figsize=(8,4.8));ax.set_xlim(-.7,6.8);ax.set_ylim(-2,2.2);ax.axis("off")
    xs=np.arange(4)*2
    for i,x in enumerate(xs):
        ax.plot(x,0,"o",color=blue,ms=8);ax.text(x,.25,f"{i}",ha="center",fontweight="bold")
        ax.plot([x,x],[-.05,-.5],color=gray)
        ax.plot([x-.22,x+.22],[-.5,-.5],color=gray,lw=2)
        ax.plot([x-.22,x+.22],[-.65,-.65],color=gray,lw=2)
        ax.plot([x,x],[-.65,-1.05],color=gray)
        for j,w in enumerate([.25,.16,.07]):ax.plot([x-w,x+w],[-1.05-j*.1]*2,color=gray)
        ax.text(x,-1.55,"100 нФ",ha="center")
        ax.annotate("",(x,.05),(x,1.2),arrowprops={"arrowstyle":"->","color":gray})
        ax.text(x,1.35,f"подготовка V{i}\nи буфер АЦП",ha="center",fontsize=9)
        if i<3:
            ax.plot([x,x+.55],[0,0],color=gray)
            ax.add_patch(Rectangle((x+.55,-.11),.9,.22,fill=False,edgecolor=gray))
            ax.plot([x+1.45,x+2],[0,0],color=gray)
            ax.text(x+1,-.35,"100 кОм",ha="center",fontsize=9)
    ax.plot([0,0,2.55],[.05,.85,.85],color=orange,ls="--",lw=1.8)
    ax.plot([3.45,6,6],[.85,.85,.05],color=orange,ls="--",lw=1.8)
    ax.add_patch(Rectangle((2.55,.74),.9,.22,fill=False,edgecolor=orange,lw=1.8))
    ax.text(3,1.02,"контрольная связь 0–3: 500 кОм",ha="center",color=orange,fontsize=10,
            bbox={"facecolor":"white","edgecolor":"none","pad":1})
    ax.text(3,-1.93,"Общий провод; нагрузка каждого буфера в расчёте 1 ГОм",ha="center",fontsize=9)
    savefig(fig,"rc_circuit")
    # Time series and the full/partial heat trace show distinct measured objects.
    nominal_label=next(k for k,v in truth["samples"].items() if v["meaning"]=="nominal local network")
    times,V=read_sample(nominal_label,data_dir)
    L=laplacian(np.ones(3));E=evolution(L,times)
    fig,axs=plt.subplots(1,2,figsize=(9,3.9))
    for i in range(4):axs[0].plot(times*1000,V[:,i,0],label=f"узел {i}",lw=1)
    axs[0].set_xlim(0,100);axs[0].set_xlabel("Время, мс");axs[0].set_ylabel("Напряжение, В");axs[0].legend(fontsize=8)
    axs[0].set_title("Синтетическая запись, подготовка 0",fontsize=10)
    axs[1].plot(times*1000,np.trace(E,axis1=1,axis2=2)/4,label="полный след / 4",color=blue)
    axs[1].plot(times*1000,(E[:,0,0]+E[:,3,3])/2,label="два крайних возврата / 2",color=orange,ls="--")
    axs[1].set_xlim(0,80);axs[1].set_ylim(.23,1.02);axs[1].set_xlabel("Время, мс");axs[1].set_ylabel("Нормированный возврат")
    axs[1].set_title("Полный и частичный доступ",fontsize=10);axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.15)
    fig.tight_layout();savefig(fig,"rc_records_trace")
    # Shared-record noise covariance, not three independent measurements.
    corr=np.asarray(result["samples"][nominal_label]["laplace_noise_correlation"])
    fig,axs=plt.subplots(1,2,figsize=(8.5,3.5),gridspec_kw={"width_ratios":[1,1.5]})
    im=axs[0].imshow(corr,vmin=0,vmax=1,cmap="Blues")
    axs[0].set_xticks(range(3),["20","60","200"]);axs[0].set_yticks(range(3),["20","60","200"])
    axs[0].set_xlabel("s, с⁻¹");axs[0].set_ylabel("s, с⁻¹")
    for i in range(3):
        for j in range(3):axs[0].text(j,i,f"{corr[i,j]:.3f}",ha="center",va="center",color="white" if corr[i,j]>.7 else "black")
    axs[0].set_title("Корреляция ошибок",fontsize=10)
    s=np.asarray(result["calibration"]["laplace_s_per_s"])
    tails=np.asarray(result["samples"][nominal_label]["finite_tail_bound_s"])
    sig=np.asarray(result["samples"][nominal_label]["contrast_std_s"])
    quad=np.asarray(result["samples"][nominal_label]["simpson_bound_under_Hcal_s"])
    axs[1].semilogy(s,sig,"o-",label="стандартная ошибка",color=blue)
    axs[1].semilogy(s,tails,"s--",label="граница хвоста",color=orange)
    axs[1].semilogy(s,quad,"^:",label="граница квадратуры",color=gray)
    axs[1].set_xlabel("s, с⁻¹");axs[1].set_ylabel("Ошибка контраста, с");axs[1].grid(alpha=.15)
    axs[1].legend(fontsize=8);axs[1].set_title("Состав расчётной ошибки",fontsize=10)
    fig.tight_layout();savefig(fig,"rc_covariance")
    # Fixed contrast envelope reveals the preselected strong example.
    fig,ax=plt.subplots(figsize=(7.2,3.6))
    labels=result["calibration"]["source_files"]
    lo=result["contrast_lower_s"][0]*1000;hi=result["contrast_upper_s"][0]*1000
    ax.axhspan(lo,hi,color=blue,alpha=.14,label="все обычные веса ±2%")
    for k,label in enumerate(labels):
        item=result["samples"][label]
        ax.errorbar(k,item["contrast_s"][0]*1000,yerr=item["primary_total_margin_s"]*1000,
                    fmt="o",color=orange if item["Hcal_rejected_by_global_envelope"] else blue,capsize=4)
    names={"nominal local network":"эталон", "bounded local drift":"дрейф", "bounded drift plus added edge":"добавленная связь"}
    ax.set_xticks(range(3),[f"{label}\n{names[truth['samples'][label]['meaning']]}" for label in labels])
    ax.set_ylabel("Контраст aᵀG(20)a, мс")
    ax.set_xlim(-.5,2.5);ax.grid(axis="y",alpha=.15)
    ax.legend(fontsize=9);fig.tight_layout();savefig(fig,"rc_blind_results")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["generate","analyse","all"],nargs="?",default="all")
    parser.add_argument("--data-dir",type=Path,default=DATA)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    with threadpool_limits(limits=1):
        if args.action in ["generate","all"]:generate(args.data_dir)
        if args.action in ["analyse","all"]:
            result=analyse(args.data_dir)
            print(json.dumps({k:{"global_Hcal_rejection":v["Hcal_rejected_by_global_envelope"],
                    "contrast_s":v["contrast_s"][0],"local_chi2_per_observation":v["local_fit"]["chi2_per_observation"],
                    "best_added_edge":v["added_edge_fits"][0]} for k,v in result["samples"].items()},ensure_ascii=False,indent=2))
        if args.action=="all":
            validation=evaluate_and_plot(args.data_dir)
            print(json.dumps(validation,ensure_ascii=False,indent=2))
