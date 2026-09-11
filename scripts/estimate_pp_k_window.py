#!/usr/bin/env python3
"""Conditional PP origin entropy and explicit K/window model averaging.

No fitted correctness weights, no truncated-search counts, no averaging entropies.
All probabilities below are stipulated sensitivity models, not calibrated posteriors.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from estimate_pp_origin_entropy import (PAPER, SCALE, TOL, load_events,
    complete_support, temporal_parameters, write)
from estimate_pp_entropy import entropy, log_density

HORIZONS=(3,7,30,86,180,math.inf)
MODELS=('reuse_mix','fifo_exp','little_exp')
R_VALUES=(.25,.5,.75)


def k_weights(r, sampling):
    """P(K=k), k=1..3, plus the unrenormalized K>3 tail.

    For geometric K per deposit, sampling a withdrawal length-biases its K
    distribution in a fully observed population. Finite-export sampling need not
    have precisely these weights; direct geometric weighting is also reported.
    """
    if not 0<r<1:
        raise ValueError('r must be between 0 and 1')
    k=np.arange(1,4)
    if sampling=='withdrawal_size_biased':
        weights=k*(1-r)**2*r**(k-1)
    elif sampling=='direct_geometric':
        weights=(1-r)*r**(k-1)
    else:
        raise ValueError(sampling)
    return weights,float(1-weights.sum())


def delay_cdf(model, x, fit, fifo, little):
    x=np.maximum(np.asarray(x,dtype=float),0)
    if model=='reuse_mix':
        means=np.asarray(fit['means_days']); weights=np.asarray(fit['mixture_weights'])
        return np.sum(weights*(-np.expm1(-x[...,None]/means)),axis=-1)
    mean=fifo if model=='fifo_exp' else little
    return -np.expm1(-x/mean)


def conditional_components(b, age, observable, T, cdf, bits):
    """q_k(d) propto b(d)*g_k(d)*1[exact k support], k=1..3.

    g_k = 1[age<=T] F(min(T, export_end-deposit_time))**(k-1).
    Given the observed target delay, only the OTHER k-1 delays are random under
    the assumed iid delay model. This is not a measured last-withdrawal CDF.
    """
    inside=age<=T
    coverage=cdf(np.minimum(T,observable))
    components=np.zeros((3,len(b))); completion=np.zeros(3)
    n=np.zeros(3,dtype=int); h=np.full(3,np.nan); h0=np.full(3,np.nan)
    support_mass=np.zeros(3)
    for i in range(3):
        weights=b*inside*coverage**i
        completion[i]=weights.sum()
        mask=(bits & (1<<i))>0
        if np.any(mask & ~inside):
            raise AssertionError('Amount support violates the window')
        n[i]=np.count_nonzero(mask & (b>0))
        q=weights*mask; z=q.sum(); support_mass[i]=z
        if z>0:
            components[i]=q/z; h[i]=entropy(components[i]); h0[i]=math.log2(n[i])
    return components,completion,n,h,h0,support_mass


def average_components(b, components, completion, prior):
    """Conservative model averaging; unavailable components abstain to b.

    Prior component masses are NOT Bayesian evidence-updated P(K|amounts).
    Such an update requires an amount-generation/observation likelihood.
    """
    available=components.sum(axis=1)>0
    alpha=prior*completion*available
    background=1-alpha.sum()
    assert background>=-1e-12
    p=background*b+alpha@components
    assert abs(p.sum()-1)<1e-10
    return p,alpha,float(background)


def summarize(values):
    values=np.asarray(values,dtype=float); finite=np.isfinite(values)
    out={'n_defined':int(finite.sum())}
    for name,fun in [('mean',np.mean),('median',np.median),('p10',lambda x:np.quantile(x,.1)),('p90',lambda x:np.quantile(x,.9))]:
        out[name]=float(fun(values[finite])) if finite.any() else math.nan
    return out


def run(source,paper,output):
    output.mkdir(parents=True,exist_ok=True)
    reuse,fit,fifo=temporal_parameters(paper)
    conditional_summary=[]; mixture_summary=[]; scenarios={}
    conditional_fields=['scenario','model','T_days','withdrawal_index','K','n_origins',
        'H_conditional','H0_origins','completion_mass','support_mass','H_uniform','H_time',
        'H_feasible','time_mass_inside','feasible_mass_inside','followup_days']
    mixture_fields=['scenario','model','T_days','withdrawal_index','sampling','r',
        'H_mixture','H_uniform','H_time','H_feasible','alpha1','alpha2','alpha3',
        'background_mass','K_gt3_prior_mass','incomplete_prior_mass','unsupported_prior_mass']
    with tempfile.TemporaryDirectory(prefix='pp-k-window-') as dirname, \
         gzip.open(output/'conditional_per_withdrawal.csv.gz','wt',newline='') as cf, \
         gzip.open(output/'mixture_per_withdrawal.csv.gz','wt',newline='') as mf:
        temp=Path(dirname); exe=temp/'support'
        subprocess.run(['c++','-O3','-std=c++17',str(Path(__file__).with_name('pp_complete_support.cpp')),'-o',str(exe)],check=True)
        cw=csv.DictWriter(cf,fieldnames=conditional_fields); cw.writeheader()
        mw=csv.DictWriter(mf,fieldnames=mixture_fields); mw.writeheader()
        for scenario,dedup in [('exact_rows_collapsed',True),('as_exported',False)]:
            deposits,withdrawals,meta=load_events(source,dedup)
            dt=np.array([t for t,a in deposits]); da=np.array([a for t,a in deposits])
            wt=np.array([t for t,a in withdrawals]); wa=np.array([a for t,a in withdrawals])
            end=meta['shared_end_day']
            little=(sum(a*(end-t) for t,a in deposits)-sum(a*(end-t) for t,a in withdrawals))/sum(a for t,a in withdrawals)
            if little<=0: raise ValueError('Nonpositive Little scale')
            meta['little_mean_days']=little; scenarios[scenario]=meta
            # Target-local arrays avoid repeated fitting/normalization per horizon.
            priors={model:[] for model in MODELS}
            for model in MODELS:
                for j,t in enumerate(wt):
                    eligible=dt<t; age=t-dt[eligible]
                    logs=log_density(model,age,fit,fifo,little)
                    p=np.exp(logs-logs.max()); p/=p.sum()
                    feasible=da[eligible]+TOL>=wa[j]
                    b=p*feasible
                    if b.sum()==0: raise ValueError('No feasible origin')
                    b/=b.sum()
                    priors[model].append((eligible,age,end-dt[eligible],p,b))
            for T in HORIZONS:
                label='all_observed' if math.isinf(T) else str(T)
                print(f'{scenario}: searching T={label}',flush=True)
                bits=complete_support(deposits,withdrawals,exe,temp,horizon=T)
                for model in MODELS:
                    conditional=[]; mixed=[]
                    cdf=lambda x:delay_cdf(model,x,fit,fifo,little)
                    for j,t in enumerate(wt):
                        eligible,age,observable,p,b=priors[model][j]
                        components,completion,n,h,h0,support_mass=conditional_components(b,age,observable,T,cdf,bits[eligible,j])
                        uniform=math.log2(len(p)); ht=entropy(p); hb=entropy(b)
                        common=dict(scenario=scenario,model=model,T_days=label,withdrawal_index=j,
                                    H_uniform=uniform,H_time=ht,H_feasible=hb)
                        for i in range(3):
                            row=dict(common,K=i+1,n_origins=int(n[i]),H_conditional=h[i],H0_origins=h0[i],
                                completion_mass=completion[i],support_mass=support_mass[i],
                                time_mass_inside=float(p[age<=T].sum()),feasible_mass_inside=float(b[age<=T].sum()),
                                followup_days=end-t)
                            cw.writerow(row); conditional.append(row)
                        for sampling in ('withdrawal_size_biased','direct_geometric'):
                            for r in R_VALUES:
                                prior,tail=k_weights(r,sampling)
                                pm,alpha,bg=average_components(b,components,completion,prior)
                                incomplete=float(np.dot(prior,1-completion))
                                unsupported=float(np.dot(prior*completion,components.sum(axis=1)==0))
                                assert abs(bg-tail-incomplete-unsupported)<1e-10
                                row=dict(common,sampling=sampling,r=r,H_mixture=entropy(pm),alpha1=alpha[0],alpha2=alpha[1],alpha3=alpha[2],
                                    background_mass=bg,K_gt3_prior_mass=tail,incomplete_prior_mass=incomplete,unsupported_prior_mass=unsupported)
                                mw.writerow(row); mixed.append(row)
                    for k in (1,2,3):
                        rr=[r for r in conditional if r['K']==k]
                        for cohort in ('all','matched','matched_followup_90d'):
                            cc=[r for r in rr if cohort=='all' or (r['n_origins']>0 and (cohort=='matched' or r['followup_days']>=90))]
                            for metric in ('H_conditional','H0_origins','H_uniform','H_time','H_feasible','completion_mass','time_mass_inside'):
                                conditional_summary.append(dict(scenario=scenario,model=model,T_days=label,K=k,cohort=cohort,metric=metric,
                                    n=len(cc),n_no_match=sum(r['n_origins']==0 for r in cc),n_singleton=sum(r['n_origins']==1 for r in cc),
                                    **summarize([r[metric] for r in cc])))
                    for sampling in ('withdrawal_size_biased','direct_geometric'):
                        for r in R_VALUES:
                            rr=[x for x in mixed if x['sampling']==sampling and x['r']==r]
                            for metric in ('H_mixture','H_uniform','H_time','H_feasible','background_mass','alpha1','alpha2','alpha3','K_gt3_prior_mass','incomplete_prior_mass','unsupported_prior_mass'):
                                mixture_summary.append(dict(scenario=scenario,model=model,T_days=label,sampling=sampling,r=r,metric=metric,n=len(rr),**summarize([x[metric] for x in rr])))
                            for reference in ('H_time','H_feasible'):
                                mixture_summary.append(dict(scenario=scenario,model=model,T_days=label,sampling=sampling,r=r,metric='paired_reduction_from_'+reference,n=len(rr),**summarize([x[reference]-x['H_mixture'] for x in rr])))
                print(f'{scenario}: T={label} complete',flush=True)
    write(output/'conditional_summary.csv',conditional_summary)
    write(output/'mixture_summary.csv',mixture_summary)
    metadata={
        'support':'Origin positive deposit records for every withdrawal in shared export coverage, not combinations or people. Exact 1/2/3-withdrawal support is enumerated uncapped separately for every T; sizes can overlap for an origin.',
        'horizon':'T is the maximum delay from deposit to ANY withdrawal in the closed-spend decomposition. It implies the deposit is within T before the target, and also constrains other withdrawals. All-observed removes T but not the export endpoint.',
        'baseline':'b(d) is the time-density prior conditioned on deposit amount sufficient for the target. H_uniform and H_time separately retain all preceding positive deposit records.',
        'k_prior':'Per-deposit geometric P(K=k)=(1-r)r^(k-1). Main sampling scenario uses k*(1-r)^2*r^(k-1), the size-biased distribution for a randomly sampled withdrawal in a fully observed population. Finite-export sampling is not exactly corrected by this formula; direct geometric target weights are sensitivity only.',
        'completion':'Given target age s and total K, other K-1 delays assumed iid from the chosen descriptive density, so g_k(d)=1[s<=T]*F(min(T,end-td))^(k-1). K=1 has no unobserved sibling delay. Reuse delays are not validated complete-spend lifetimes; common latent user timing and within-deposit dependence are not fitted.',
        'conditional':'c_k=sum_d b(d)g_k(d); q_k(d) proportional to b(d)g_k(d)1[d supports an exact k decomposition]. Strict H(q_k) undefined on empty support. H0 separately counts supported origins, not withdrawal combinations.',
        'mixture':'alpha_k=pi_k*c_k if q_k exists, else 0. p_mix=(1-sum alpha_k)*b+sum alpha_k*q_k. Compute entropy AFTER marginalizing K, since origins can overlap. alpha are stipulated pre-evidence regime masses, not posterior P(K|observed amounts). Fitting those posterior weights needs an amount-generation/observation likelihood; this procedure does not supply one.',
        'background':'Mass for K>3, incomplete/outside-window histories and unsupported components returns to broad feasible prior b. b includes both inside and outside candidates. This is explicit conservative abstention/model averaging, NOT a generative posterior conditioned on the complement event and NOT a guaranteed upper bound on entropy.',
        'matching':'Gross withdrawal eth_amount summed by transaction hash, tolerance 1e-9 ETH, fixed-point 1e-12 ETH. Positive deposit records only; exact row collapse vs raw exports both reported. Zero initial balance and single-origin no-balance-increase interpretation are assumptions.',
        'retrospective':'Search may use withdrawals after the target up to the shared export endpoint. Finite follow-up is represented in g_k and in observed support but this is not a calibrated censoring correction.',
        'reuse_selection':'Reuse fit is selected sample only; FIFO and Little supply assumed exponential scales, not independently fitted lifetime distributions. No verified funding labels or posterior privacy entropy.',
        'r_values':list(R_VALUES),'horizons_days':[str(t) for t in HORIZONS],'scenarios':scenarios,'reuse_fit':fit,'fifo_mean_days':fifo,
        'reuse_coverage':{str(t):float(np.mean(reuse<=t)) for t in HORIZONS}}
    paths=[Path(__file__),Path(__file__).with_name('pp_complete_support.cpp'),Path(__file__).with_name('estimate_pp_origin_entropy.py'),Path(__file__).with_name('estimate_pp_entropy.py'),
        source/'data/processed/processed_privacypools_eth_pool_deposits.csv',source/'data/processed/processed_privacypools_data_withdraws.csv',
        paper/'Figures/pp_h1_pairs.csv',paper/'Figures/figure_ch4_03_pp_retained_fifo.csv']
    metadata['sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (output/'model_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    render(output,conditional_summary,mixture_summary,metadata)


def render(output,cs,ms,metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def c(T,k,metric='H_conditional',cohort='all'):
        return next(r for r in cs if r['scenario']=='exact_rows_collapsed' and r['model']=='reuse_mix' and r['T_days']==str(T) and r['K']==k and r['metric']==metric and r['cohort']==cohort)
    def m(T,r,metric='H_mixture',model='reuse_mix',sampling='withdrawal_size_biased'):
        return next(x for x in ms if x['scenario']=='exact_rows_collapsed' and x['model']==model and x['T_days']==str(T) and x['r']==r and x['metric']==metric and x['sampling']==sampling)
    lines=['# PP: feltételes K–ablak entrópia és modellátlagolás','',
        'A feltételes eredmények adott K-ra és teljes, T-n belüli felhasználásra vonatkoznak. A kevert eredmény a K>3, az ablakon kívüli/befejezetlen és az egzakt támogatás nélküli komponenseket is megtartja. Mindkettő feltételezett modell entrópiája; egyik sem validált funding posterior.', '',
        '## Feltételes eredmények: reuse-időmodell, azonos sorok összevonva', '',
        '| T (nap) | K | találatos / összes kifizetés | egyetlen eredetjelölt | feltételes medián (bit) |',
        '|---|---:|---:|---:|---:|']
    for T in ('3','7','30','86','180','all_observed'):
        for k in (1,2,3):
            x=c(T,k)
            lines.append(f"| {T} | {k} | {x['n_defined']} / {x['n']} | {x['n_singleton']} | {x['median']:.3f} |")
    lines += ['', 'A feltételes medián csak nemüres támogatáson definiált. A nevező végig ugyanaz, de a találatos részhalmaz T-vel és K-val változik; a mediánok különbsége nem párosított entrópiacsökkenés.', '',
        '## Kevert eredmények: minden kifizetés, méret szerint súlyozott K-prior', '',
        '| T | r=.25 | r=.5 | r=.75 | háttér átlagos tömege (r=.5) |',
        '|---|---:|---:|---:|---:|']
    for T in ('3','7','30','86','180','all_observed'):
        lines.append('| '+T+' | '+' | '.join(f"{m(T,r)['median']:.3f}" for r in R_VALUES)+f" | {m(T,.5,'background_mass')['mean']:.1%} |")
    lines += ['', 'Az egyenletes alapmodell mediánja '+f"{m('30',.5,'H_uniform')['median']:.3f}"+' bit, a reuse-időmodellé '+f"{m('30',.5,'H_time')['median']:.3f}"+' bit, az elegendő befizetésösszegre is feltételes időmodellé '+f"{m('30',.5,'H_feasible')['median']:.3f}"+' bit. A kevert modell ez utóbbihoz adja az egzakt felbontásokat; a különbségek párosított eloszlása is szerepel a CSV-ben.', '',
        '## Időmodell-érzékenység (r=.5)', '',
        '| Időmodell | T=30 medián | teljes megfigyelés medián |', '|---|---:|---:|']
    for model in MODELS:
        lines.append(f"| {model} | {m('30',.5,model=model)['median']:.3f} | {m('all_observed',.5,model=model)['median']:.3f} |")
    lines += ['', '## A súlyok jelentése', '',
        '- Befizetésenként geometriai K: P(K=k)=(1-r)r^(k-1). Kifizetéseket vizsgálva a teljesen megfigyelt populációban méret szerinti súlyozás kell: pi_k=k(1-r)^2 r^(k-1). A véges export nem ilyen teljes minta; a közvetlen geometriai súlyozást ezért külön érzékenységi változatként közöljük.',
        '- A K>3 tömeg nincs visszanormalizálva az első három méretre. r=.5 mellett ez a tömeg a fő változatban 31,25%, a közvetlen geometriai változatban 12,5%.',
        '- A T ablak a befizetéstől az összes felbontási kifizetésig tart. A már megfigyelt célkifizetés mellett K-1 további késleltetés marad. Független, azonos eloszlású késleltetéseket feltételezve a megfigyelhető teljesség tényezője F(min(T, exportvége-befizetés))^(K-1), ha a célkifizetés is T-n belül van. Ez nem empirikusan becsült utolsókifizetés-eloszlás.',
        '- A b(d) alapeloszlás minden korábbi, elegendő összegű befizetést megtart, az időpriorral súlyozva. q_k(d) ezt súlyozza a teljességi tényezővel és az adott K egzakt támogatásával.',
        '- A keverési súly pi_k szorozva a b szerinti átlagos teljességi tényezővel. Üres támogatásnál az egész komponens visszakerül a b háttéreloszlásba. Nem szorozzuk az entrópiát a komponens valószínűségével: előbb a befizetésjelöltek eloszlását keverjük.',
        '- A súlyok modellátlagolási feltevések, nem az összegek megfigyelése után Bayes-szel frissített K-valószínűségek. Ehhez a kifizetési összegek generatív modellje hiányzik. A háttérhez visszatérés tartózkodás, nem az ablakon kívüliségre szigorúan feltételes posterior.', '',
        '## Megmaradó korlátok', '',
        'A felbontások teljes keresése csak 1–3 kifizetésre teljes. A feldarabolás és a késleltetések közötti függést nem tanultuk meg. A reuse-minta szelektált, FIFO és Little pedig feltételezett időskálát ad. A knapsack retrospektív: a célkifizetés utáni megfigyeléseket is látja. Nincsenek igazolt eredetbefizetés-címkék vagy logszintű depositazonosítók. Az egyes célkifizetések eloszlásait külön számoljuk; globális, kettős felhasználást kizáró történetmodellt nem illesztünk.', '',
        'A nyers és az azonos sorokat összevonó változat, a közvetlen és méret szerint súlyozott K-prior, az összes időmodell és a párosított csökkenések a CSV-kben megtalálhatók.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4.1))
    x=np.arange(6); labels=['3','7','30','86','180','All observed']
    for k in (1,2,3):
        axes[0].plot(x,[c(T,k)['median'] for T in ('3','7','30','86','180','all_observed')],marker='o',label=f'K={k}')
    for r in R_VALUES:
        axes[1].plot(x,[m(T,r)['median'] for T in ('3','7','30','86','180','all_observed')],marker='o',label=f'r={r}')
    axes[1].axhline(m('30',.5,'H_time')['median'],ls='--',color='black',label='Time only')
    axes[1].axhline(m('30',.5,'H_feasible')['median'],ls=':',color='gray',label='Time + sufficient amount')
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8); ax.set_xlabel('Post-deposit horizon (days)'); ax.set_ylabel('Median origin entropy (bits)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_title('Conditional on K and exact support\nMatched subsets vary across points',fontsize=10)
    axes[1].set_title('K/window mixture; all withdrawals\nSize-biased geometric prior',fontsize=10)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'k_window_entropy.{ext}',dpi=180)
    plt.close(fig)
    tex=[r'% Generated by estimate_pp_k_window.py; reuse time model; exact rows collapsed.',
        r'\begin{tabular}{lrrr}',r'\hline',r'$T$ (days) & $r=.25$ & $r=.5$ & $r=.75$ \\',r'\hline']
    for T in ('3','7','30','86','180','all_observed'):
        tex.append(('Observed history' if T=='all_observed' else T)+' & '+' & '.join(f"{m(T,r)['median']:.2f}" for r in R_VALUES)+r' \\')
    tex += [r'\hline',r'\end{tabular}']
    (output/'k_window_table.tex').write_text('\n'.join(tex)+'\n')
    tex=[r'% Generated; reuse time model, exact rows collapsed, T=30 days.',
        r'\begin{tabular}{rrrr}',r'\hline',r'$K$ & Matched targets & Time & Conditional \\',r'\hline']
    for k in (1,2,3):
        cc=c('30',k);tt=c('30',k,'H_time','matched')
        tex.append(f"{k} & {cc['n_defined']:,} & {tt['median']:.2f} & {cc['median']:.2f}"+r' \\')
    tex += [r'\hline',r'\end{tabular}']
    (output/'conditional_30days_table.tex').write_text('\n'.join(tex)+'\n')
    stats=(f"The common-coverage analysis contains {metadata['scenarios']['exact_rows_collapsed']['withdrawal_transactions']:,} withdrawal transactions and "
           f"{metadata['scenarios']['exact_rows_collapsed']['deposit_records']:,} positive deposit records after exact-row collapse. "
           f"Median origin entropy is {m('30',.5,'H_uniform')['median']:.2f} bits under the uniform prior and "
           f"{m('30',.5,'H_time')['median']:.2f} bits under reuse-delay weighting. "
           f"With the sufficient-amount restriction alone it is {m('30',.5,'H_feasible')['median']:.2f} bits. "
           f"For the size-biased geometric model with $r=0.5$, the mixture median is {m('30',.5)['median']:.2f} bits at $T=30$ days "
           f"and {m('all_observed',.5)['median']:.2f} bits over the observed history. "
           "These are entropies of stipulated distributions, not validated posterior privacy estimates.\n")
    (output/'statistics.tex').write_text('% Generated by estimate_pp_k_window.py\n'+stats)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=PAPER.parent/'privacypools-deanonymization-main')
    ap.add_argument('--paper',type=Path,default=PAPER);ap.add_argument('--output',type=Path,default=PAPER/'analysis/pp_k_window')
    args=ap.parse_args();run(args.source.resolve(),args.paper.resolve(),args.output.resolve())
