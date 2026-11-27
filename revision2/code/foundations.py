"""Exact introductory examples for the second edition. All data are synthetic."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
RESULTS = ROOT / 'results'
FIG.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})

def laplacian(weights):
    L=np.zeros((3,3))
    for (i,j),w in weights.items():
        b=np.eye(3)[i]-np.eye(3)[j]
        L+=w*np.outer(b,b)
    return L

def run():
    a={(0,1):1.,(1,2):1.}
    b={(0,1):1/3,(1,2):1/3,(0,2):4/3}
    LA=laplacian(a);LB=laplacian(b)
    t=np.geomspace(.015,20.,350)
    evA=np.linalg.eigvalsh(LA);evB=np.linalg.eigvalsh(LB)
    e=np.exp(-t[:,None]*np.array([0.,1.,3.])[None,:])
    P=e.mean(axis=1)
    ds=2*t*(e[:,1]+3*e[:,2])/e.sum(axis=1)
    K_A=1/3+2/3*np.exp(-3*t)
    K_B=1/3+2/3*np.exp(-t)
    GA=np.linalg.inv(LA+np.eye(3));GB=np.linalg.inv(LB+np.eye(3))
    sample=1.
    exactA=expm(-LA*sample);exactB=expm(-LB*sample)
    payload={
        'status':'Analytic finite graph and sphere examples; no physical measurements',
        'node_order':[0,1,2], 'ordinary_rate_per_second':1.,
        'L_A_per_second':LA.tolist(),'L_B_per_second':LB.tolist(),
        'eigenvalues_A_per_second':evA.tolist(),'eigenvalues_B_per_second':evB.tolist(),
        'G_A_at_s1_seconds':GA.tolist(),'G_B_at_s1_seconds':GB.tolist(),
        'local_heat_at_t1':{'A':float(exactA[1,1]),'B':float(exactB[1,1]),
                            'difference':float(exactB[1,1]-exactA[1,1])},
        'trace_at_t1':float((1+np.exp(-1)+np.exp(-3))/3),
        'ds_at_t1':float(2*(np.exp(-1)+3*np.exp(-3))/(1+np.exp(-1)+np.exp(-3))),
        'finite_band_seconds':[.015,20.],
        'complete_trace_equality_error':float(np.max(np.abs(np.exp(-t[:,None]*evA).mean(axis=1)-np.exp(-t[:,None]*evB).mean(axis=1)))),
        'sphere':{'R_m':10.,'r_m':1.,'C_m':float(20*np.pi*np.sin(.1)),
                  'flat_C_m':float(2*np.pi),'difference_m':float(2*np.pi-20*np.pi*np.sin(.1)),
                  'exact_curvature_per_m2':.01,
                  'leading_estimator_per_m2':float(6*(1-10*np.sin(.1)))}
    }
    (RESULTS/'foundations.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))

    fig,axs=plt.subplots(2,2,figsize=(9,6.3),gridspec_kw={'height_ratios':[.9,1.25]})
    for ax,edges,label in [(axs[0,0],a,'A  Цепь'),(axs[0,1],b,'B  Треугольник')]:
        pos={0:np.array([0.,0.]),1:np.array([1.,.58]),2:np.array([2.,0.])}
        for (i,j),w in edges.items():
            pi,pj=pos[i],pos[j]
            ax.plot([pi[0],pj[0]],[pi[1],pj[1]],color='#6b7780',lw=2)
            mid=(pi+pj)/2
            txt='1' if w==1 else ('1/3' if w<1 else '4/3')
            ax.text(mid[0],mid[1]+(.07 if (i,j)!=(0,2) else -.12),txt,ha='center',va='center',fontsize=10)
        for i,xy in pos.items():
            ax.scatter(*xy,s=300,color='#b5653b' if i==1 else '#336c7b',zorder=4)
            ax.text(xy[0],xy[1],str(i),ha='center',va='center',color='white',fontsize=10,zorder=5)
        ax.set_xlim(-.25,2.25);ax.set_ylim(-.25,.9);ax.axis('off');ax.set_title(label,fontsize=12)
    ax=axs[1,0]
    ax.semilogx(t,ds,color='#336c7b',lw=3,label='A')
    ax.semilogx(t,ds,color='#b5653b',ls='--',lw=1.5,label='B')
    ax.set_xlabel('Время диффузии t, с');ax.set_ylabel('Спектральный показатель dₛ')
    ax.set_title('Совпадение на всей оси t');ax.legend(frameon=False);ax.grid(alpha=.15)
    ax=axs[1,1]
    ax.semilogx(t,K_A,color='#336c7b',lw=2,label='A')
    ax.semilogx(t,K_B,color='#b5653b',lw=2,label='B')
    ax.set_xlabel('Время после импульса t, с');ax.set_ylabel('Локальный возврат K₁₁(t)')
    ax.set_title('Различие в выбранном узле');ax.legend(frameon=False);ax.grid(alpha=.15)
    fig.tight_layout(pad=1.5)
    for ext in ['png','pdf']:fig.savefig(FIG/f'foundations_isospectral.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)

    x=np.linspace(.01,np.pi-.015,300)
    area_ratio=2*(1-np.cos(x))/(x*x)
    dimension=x/np.tan(x/2)
    fig,axs=plt.subplots(1,2,figsize=(8.7,3.4))
    axs[0].plot(x,area_ratio,color='#336c7b',lw=2)
    axs[0].set_ylabel('Площадь диска / (πr²)')
    axs[0].set_title('Геодезический диск на сфере')
    axs[1].plot(x,dimension,color='#b5653b',lw=2)
    axs[1].axhline(2,color='#6b7780',ls=':',lw=1)
    axs[1].set_ylabel('Объёмный показатель dᵥ')
    axs[1].set_title('Локальная размерность остаётся 2')
    for ax in axs:ax.set_xlabel('r/R');ax.grid(alpha=.15)
    fig.tight_layout(pad=1.3)
    for ext in ['png','pdf']:fig.savefig(FIG/f'foundations_sphere.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
