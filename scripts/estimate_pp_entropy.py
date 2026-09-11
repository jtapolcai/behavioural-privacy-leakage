#!/usr/bin/env python3
"""Finite-observation PP entropy comparison, conditional on combination size.

Time-only universe: all distinct k-subsets of later observed withdrawals.
Amount-conditioned universe: the reconciled matching export, assumed exhaustive
only where its truncated flag is zero. No funding posterior calibration claimed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.special import logsumexp
from scipy.integrate import trapezoid

from refresh_pp_measurements import check_details, COUNTS, PREFIXES, SUFFIXES, read_csv, write_csv

PAPER = Path(__file__).resolve().parents[1]
HORIZONS = (3, 7, 30, 86, 180)


def decimal_amount(value):
    value=value.strip()
    if ',' in value and '.' in value:
        raise ValueError('Ambiguous decimal/thousands separators')
    return Decimal(value.replace(',','.'))


def day(value):
    dt = datetime.fromisoformat(value.strip().replace(' UTC', '+00:00').replace(',', '.'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()/86400


def entropy(weights):
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if not len(weights) or total <= 0:
        return math.nan
    prob = weights[weights > 0]/total
    return float(-np.sum(prob*np.log2(prob)))


def subset_prefix(weights, k):
    """Z and sum(weight*log(weight)) for k-subsets of every prefix, O(n*k).

    A combination has weight product(v_i). Recurrence enumerates its last
    member implicitly, so no combinatorial-size time-only list is materialized.
    """
    weights = np.asarray(weights, dtype=float)
    logw = np.zeros_like(weights)
    np.log(weights, out=logw, where=weights > 0)
    prev_z = np.ones(len(weights)+1)
    prev_l = np.zeros(len(weights)+1)
    for _ in range(k):
        z_terms = weights*prev_z[:-1]
        l_terms = weights*(prev_l[:-1] + prev_z[:-1]*logw)
        prev_z = np.r_[0., np.cumsum(z_terms)]
        prev_l = np.r_[0., np.cumsum(l_terms)]
    return prev_z, prev_l


def from_partition(z, l):
    return float((math.log(z)-l/z)/math.log(2)) if z > 0 else math.nan


def coarsened_entropy(h_inside, mass_inside):
    """Retain individual inside candidates; merge all observed outside into one."""
    q = float(np.clip(mass_inside, 0, 1))
    if q == 0:
        return 0.0  # all probability in the outside category, not identified flow
    if q == 1:
        return h_inside
    return -q*math.log2(q)-(1-q)*math.log2(1-q)+q*h_inside


def fit_exp_mixture(delays):
    """Two-exponential marginal mixture, multi-start EM, descriptive fit only."""
    x = np.asarray(delays)
    assert np.all(x > 0)
    best = None
    for frac in (.2, .5, .8):
        for short, long in ((.1, 2.), (.4, 5.), (.02, 1.5)):
            means = np.array([max(x.mean()*short, 1e-6), x.mean()*long])
            pi = np.array([frac, 1-frac])
            old = -math.inf
            for _ in range(2000):
                logterms = np.log(pi)[None,:]-np.log(means)[None,:]-x[:,None]/means[None,:]
                norm = logsumexp(logterms, axis=1)
                ll = float(norm.sum())
                resp = np.exp(logterms-norm[:,None])
                size = resp.sum(axis=0)
                means = np.maximum((resp*x[:,None]).sum(axis=0)/size, 1e-6)
                pi = size/len(x)
                if abs(ll-old) < 1e-9:
                    break
                old = ll
            if best is None or ll > best[0]:
                order = np.argsort(means)
                best = (ll, means[order], pi[order])
    ll, means, pi = best
    return {'means_days': means.tolist(), 'mixture_weights': pi.tolist(),
            'sample_n': len(x), 'sample_mean_days': float(x.mean()),
            'sample_median_days': float(np.median(x)), 'log_likelihood': ll,
            'fit_scope': 'Selected address-reuse pair delays; uncensored descriptive fit, not a population holding-time estimate.'}


def log_density(model, delays, fit, fifo, little):
    if model == 'uniform':
        return np.zeros_like(delays)
    if model == 'reuse_mix':
        means = np.array(fit['means_days'])
        pi = np.array(fit['mixture_weights'])
        return logsumexp(np.log(pi)[None,:]-np.log(means)[None,:]-delays[:,None]/means[None,:], axis=1)
    mean = fifo if model == 'fifo_exp' else little
    return -math.log(mean)-delays/mean


def main(source, output, paper):
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    def load(path):
        inputs[str(path.relative_to(source) if path.is_relative_to(source) else path.relative_to(paper))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read_csv(path)
    # Transaction-hash aggregation follows existing PP matching exports.
    raw = load(source/'data/processed/processed_privacypools_data_withdraws.csv')
    agg = {}
    for r in raw:
        h = r['tx_hash']; t = day(r['evt_block_time'])
        if h in agg:
            assert agg[h]['time'] == t, 'Ambiguous withdrawal time'
            agg[h]['amount'] += decimal_amount(r['eth_amount'])
        else:
            agg[h] = {'time': t, 'amount': decimal_amount(r['eth_amount'])}
    hashes = sorted(agg, key=lambda h: (agg[h]['time'], h))
    wi = {h: i for i,h in enumerate(hashes)}
    wt = np.array([agg[h]['time'] for h in hashes])
    end = float(wt.max())
    reuse = load(paper/'Figures/pp_h1_pairs.csv')
    delays = np.array([float(r['dt_days']) for r in reuse if float(r['dt_days']) > 0])
    fit = fit_exp_mixture(delays)
    fifo_rows = load(paper/'Figures/figure_ch4_03_pp_retained_fifo.csv')
    fifo = float(np.median([float(r['fifo_horizontal_lag_days']) for r in fifo_rows if r['fifo_horizontal_lag_days']]))
    # Separate window for aggregate Little-style ratio: common export coverage.
    flows = load(paper/'Figures/figure_ch4_03_pp_cumulative_pool_flow.csv')
    deposits_raw = load(source/'data/processed/processed_privacypools_eth_pool_deposits.csv')
    common_end = min(max(day(r['time']) for r in deposits_raw), end)
    flows = [r for r in flows if day(r['date'])+1 <= common_end]
    seconds = np.array([day(r['date']) for r in flows])
    retained = np.array([float(r['retained']) for r in flows])
    elapsed = float(seconds[-1]-seconds[0])
    assert elapsed > 0
    avg_balance = float(trapezoid(retained, seconds)/elapsed)
    avg_outflow = (float(flows[-1]['cum_withdraw'])-float(flows[0]['cum_withdraw']))/elapsed
    little = avg_balance/avg_outflow
    assert min(fifo, little) > 0
    metadata = {'schema_version':1, 'reuse_fit':fit,
        'fifo_exp_mean_days':fifo, 'fifo_note':'Exponential sensitivity model sets its mean to the archived median FIFO lag; this is not a fitted delay distribution.',
        'little_exp_mean_days':little, 'little_inputs':{'start':flows[0]['date'],'end':flows[-1]['date'],
          'time_average_retained_eth':avg_balance,'average_outflow_eth_per_day':avg_outflow},
        'little_note':'Ratio of average exported balance to average outflow over common coverage. Growing pool and export duplicates invalidate a calibrated steady-state interpretation.',
        'withdrawal_rows':len(raw),'withdrawal_hashes':len(hashes),
        'observation_end_utc':datetime.fromtimestamp(end*86400,timezone.utc).isoformat(),
        'model':'Conditional on k, prior weight of subset U is product of per-withdrawal marginal delay densities; independence is an explicit approximation. No cross-k probabilities assigned.',
        'universe':'All k-subsets of observed later withdrawal transactions, with no amount filter. Match posterior conditions this finite time prior on membership in the exported amount-compatible set.',
        'outside':'q is computed from all observed future candidates. Beyond the export endpoint and unmodelled histories are not assigned calibrated mass. Coarsened entropy merges observed candidates outside T into one category.',
        'completeness':'Export flags assumed accurate. Capped targets excluded from headline posterior summaries. No-match has no amount posterior; time-only entropy remains available.',
        'cohort':'Main fixed cohort: follow-up >=180 days and non-capped. Within each horizon, before/after are paired on targets with at least one match. A stricter common_matched cohort has a match within 3 days.',
        'horizons_days':list(HORIZONS), 'per_k':{}, 'sample_reuse_coverage':{str(t):float(np.mean(delays<=t)) for t in HORIZONS}}
    all_rows = []
    models = ('uniform','reuse_mix','fifo_exp','little_exp')
    for k in (1,2,3):
        folder = source/f'results/data/Heuristics/Knapsack/{k}-sum'
        prefix = PREFIXES[k]
        counter = load(folder/f'{prefix}_{SUFFIXES[k]}.csv')
        dep = {r['dep_hash']:r for r in counter}
        details = folder/f'{prefix}_matches.csv'
        if details.exists():
            detail_bytes = details.read_bytes(); inputs[str(details.relative_to(source))] = hashlib.sha256(detail_bytes).hexdigest()
        else:
            zpath=details.with_suffix('.zip'); inputs[str(zpath.relative_to(source))] = hashlib.sha256(zpath.read_bytes()).hexdigest()
            with zipfile.ZipFile(zpath) as z:
                detail_bytes=z.read(details.name)
        check = check_details(io.StringIO(detail_bytes.decode('utf-8-sig')), counter, k)
        matched = {h:[] for h in dep}
        for r in csv.DictReader(io.StringIO(detail_bytes.decode('utf-8-sig'))):
            ids=tuple(wi[r[f'w{i}_hash']] for i in range(1,k+1))
            assert all(wt[i]>day(dep[r['dep_hash']]['dep_time']) for i in ids)
            for i,wid in enumerate(ids,1):
                # Original matching code passed amounts through float before export.
                assert math.isclose(float(agg[hashes[wid]]['amount']),float(r[f'w{i}_amount']),rel_tol=1e-12,abs_tol=1e-12)
            matched[r['dep_hash']].append(ids)
        metadata['per_k'][str(k)]={'deposits':len(counter),'capped':sum(int(r['truncated']) for r in counter),'details':check}
        for index,r in enumerate(counter):
            dtime=day(r['dep_time']); start=int(np.searchsorted(wt,dtime,side='right'))
            future=wt[start:]-dtime; future_n=len(future)
            combinations=np.array(matched[r['dep_hash']],dtype=int).reshape(-1,k)-start
            maxdel=np.max(future[combinations],axis=1) if len(combinations) else np.array([])
            horizons=(*HORIZONS, math.inf)
            common = end-dtime>=max(HORIZONS)-1e-10
            common_matched=common and bool(np.any(maxdel<=min(HORIZONS)))
            for model in models:
                logw=log_density(model,future,fit,fifo,little)
                if len(logw):logw=logw-logw.max()
                weights=np.exp(logw)
                z,l=subset_prefix(weights,k)
                match_w=np.prod(weights[combinations],axis=1) if len(combinations) else np.array([])
                z_match_full=float(match_w.sum()); z_full=float(z[-1])
                for horizon in horizons:
                    m=int(np.searchsorted(future,horizon,side='right'))
                    mask=maxdel<=horizon; selected=match_w[mask]
                    count=int(mask.sum()); zn=float(z[m]); za=float(selected.sum())
                    time_h=from_partition(zn,l[m]); amount_h=entropy(selected)
                    qt=zn/z_full if z_full>0 else math.nan
                    qa=za/z_match_full if z_match_full>0 else math.nan
                    all_rows.append(dict(k=k,target_index=index,model=model,horizon_days='all' if math.isinf(horizon) else horizon,
                        observed_followup_days=end-dtime,full_horizon_observed=int(math.isinf(horizon) or end-dtime>=horizon),
                        fixed_180d_cohort=int(common),common_matched_cohort=int(common_matched),capped=int(r['truncated']),
                        n_withdrawals=m,n_time_combinations=math.comb(m,k) if m>=k else 0,n_matching_combinations=count,
                        H_time_bits=time_h,H_amount_bits=amount_h,delta_bits=time_h-amount_h,
                        H_time_full_bits=from_partition(z_full,l[-1]),H_amount_full_bits=entropy(match_w),
                        time_mass_inside_observed=qt,amount_mass_inside_observed=qa,
                        H_time_coarse_bits=coarsened_entropy(time_h,qt) if math.isfinite(qt) else math.nan,
                        H_amount_coarse_bits=coarsened_entropy(amount_h,qa) if math.isfinite(qa) else math.nan,
                        compatible_mass_given_time=za/zn if zn>0 else math.nan))
        print(f'Completed k={k}: {len(counter)} deposits',flush=True)
    fields=list(all_rows[0])
    def cleaned(rows):
        return [{k:('' if isinstance(v,float) and not math.isfinite(v) else v) for k,v in r.items()} for r in rows]
    write_csv(output/'per_target_entropy.csv',cleaned(all_rows),fields)
    summaries=[]
    for cohort in ('fixed_180d','available','common_matched'):
        for k in (1,2,3):
            for model in models:
                for horizon in (*HORIZONS,'all'):
                    base=[r for r in all_rows if r['k']==k and r['model']==model and r['horizon_days']==horizon]
                    if cohort=='fixed_180d':base=[r for r in base if r['fixed_180d_cohort']]
                    if cohort=='common_matched':base=[r for r in base if r['common_matched_cohort']]
                    if cohort=='available':base=[r for r in base if r['full_horizon_observed']]
                    complete=[r for r in base if not r['capped']]
                    paired=[r for r in complete if r['n_matching_combinations']>0]
                    item=dict(cohort=cohort,k=k,model=model,horizon_days=horizon,
                        n_targets=len(base),n_capped=sum(r['capped'] for r in base),n_noncapped=len(complete),
                        n_matched=len(paired),n_no_match=len(complete)-len(paired),
                        n_singleton=sum(r['n_matching_combinations']==1 for r in paired),
                        fraction_entropy_increased=sum(r['delta_bits'] < -1e-8 for r in paired)/len(paired) if paired else math.nan)
                    for field in ('H_time_bits','H_amount_bits','delta_bits','H_time_full_bits','H_amount_full_bits','H_time_coarse_bits','H_amount_coarse_bits','time_mass_inside_observed','amount_mass_inside_observed','compatible_mass_given_time'):
                        vals=[r[field] for r in paired if math.isfinite(r[field])]
                        item['median_'+field]=float(np.median(vals)) if vals else math.nan
                        if field in ('H_time_bits','H_amount_bits','delta_bits'):
                            item['p10_'+field]=float(np.quantile(vals,.1)) if vals else math.nan
                            item['p90_'+field]=float(np.quantile(vals,.9)) if vals else math.nan
                    item['median_time_all_noncapped']=float(np.nanmedian([r['H_time_bits'] for r in complete])) if complete else math.nan
                    summaries.append(item)
    write_csv(output/'entropy_summary.csv',cleaned(summaries))
    # Algebraic checks on all computed rows, including capped export-only subsets.
    for r in all_rows:
        if r['model']=='uniform' and r['n_time_combinations']:
            assert math.isclose(r['H_time_bits'],math.log2(r['n_time_combinations']),abs_tol=1e-8)
        if r['model']=='uniform' and r['n_matching_combinations']:
            assert math.isclose(r['H_amount_bits'],math.log2(r['n_matching_combinations']),abs_tol=1e-8)
        if r['n_matching_combinations']==1:assert abs(r['H_amount_bits'])<1e-10
        assert r['n_matching_combinations']<=r['n_time_combinations']
        assert not math.isfinite(r['compatible_mass_given_time']) or r['compatible_mass_given_time']<=1+1e-8
    metadata['input_sha256']=inputs
    metadata['generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'model_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    # Main horizon figure holds both target population AND matched membership fixed.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),sharey=True,layout='constrained')
    for k,ax in enumerate(axes,1):
        for model,color,label in [('uniform','#777777','Uniform'),('reuse_mix','#0072b2','Reuse-delay mixture')]:
            ss=[r for r in summaries if r['cohort']=='common_matched' and r['k']==k and r['model']==model and r['horizon_days']!='all']
            x=[r['horizon_days'] for r in ss]
            ax.plot(x,[r['median_H_time_bits'] for r in ss],color=color,ls='--',marker='o',ms=3,label=label+': time only')
            ax.plot(x,[r['median_H_amount_bits'] for r in ss],color=color,marker='o',ms=3,label=label+': + amount')
        n=ss[0]['n_matched']
        ax.set(xscale='log',xlabel='Horizon (days)',title=f'1 → {k}; fixed N = {n}'+('\nExploratory: very small sample' if n<10 else ''))
        ax.set_xticks(HORIZONS,labels=[str(t) for t in HORIZONS]); ax.grid(alpha=.2)
    axes[0].set_ylabel('Paired-cohort median entropy (bits)')
    axes[0].legend(fontsize=7,loc='upper left')
    fig.savefig(output/'entropy_comparison.png',dpi=180)
    fig.savefig(output/'entropy_comparison.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    # 30-day paired comparison uses every non-capped target with 30 days observed.
    rows30=[r for r in summaries if r['cohort']=='available' and r['horizon_days']==30]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    xx=np.arange(3)
    for j,(field,label,color) in enumerate([
        ('median_H_time_bits','Time only','#999999'),
        ('median_H_amount_bits','Time + amount','#0072b2')]):
        ss=[r for r in rows30 if r['model']=='reuse_mix']
        values=[r[field] for r in ss]
        bars=axes[0].bar(xx+(j-.5)*.32,values,.32,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
    axes[0].set(xticks=xx,xticklabels=[f"1 → {r['k']}\nN = {r['n_matched']}" for r in ss],ylabel='Median conditional entropy (bits)',title='30-day horizon: same targets before / after')
    axes[0].legend(fontsize=8)
    x=np.geomspace(.01,500,300)
    means=np.array(fit['means_days']);pi=np.array(fit['mixture_weights'])
    cdf=np.sum(pi[None,:]*(1-np.exp(-x[:,None]/means[None,:])),axis=1)
    sorted_delays=np.sort(delays)
    axes[1].step(sorted_delays,np.arange(1,len(delays)+1)/len(delays),where='post',label='Address-reuse sample',color='#222222')
    axes[1].plot(x,cdf,label='Two-exponential fit',color='#0072b2')
    axes[1].plot(x,1-np.exp(-x/fifo),label=f'FIFO-scale exponential ({fifo:.0f} d)',color='#e69f00',ls='--')
    axes[1].plot(x,1-np.exp(-x/little),label=f'Little-scale exponential ({little:.0f} d)',color='#009e73',ls=':')
    axes[1].set(xscale='log',xlabel='Delay (days)',ylabel='CDF',ylim=(0,1.02),title='Time models: different sampling assumptions')
    axes[1].legend(fontsize=7);axes[1].grid(alpha=.2)
    fig.savefig(output/'entropy_30days.png',dpi=180)
    fig.savefig(output/'entropy_30days.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    tex=[r'% Generated by scripts/estimate_pp_entropy.py',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',
         r'Model & Paired $N$ & Time only & Time + amount & Full-history amount\\',r'\midrule']
    for r in rows30:
        if r['model']=='reuse_mix':
            tex.append(f"$1\\to {r['k']}$ & {r['n_matched']} & {r['median_H_time_bits']:.2f} & {r['median_H_amount_bits']:.2f} & {r['median_H_amount_full_bits']:.2f}\\\\")
    tex += [r'\bottomrule',r'\end{tabular}']
    (output/'entropy_30days_table.tex').write_text('\n'.join(tex)+'\n')
    lines=['# PP entrópiabecslés: első számítás',
           '', 'Újrafuttatás a cikk mappájából: `MPLCONFIGDIR=/tmp/pp-matplotlib /usr/bin/python3 scripts/estimate_pp_entropy.py`.',
           '', '## 30 napos eredmény',
           '', 'Címújrahasználati késleltetésekre illesztett két-exponenciális keverék. A két entrópiát ugyanazon találatos befizetéseken hasonlítjuk össze, teljes 30 napos megfigyeléssel és nem levágott exportkereséssel. Az összegfeltétel díj-/nettó konvencióját a meglévő találatexport adja.',
           '', '| Modell | Találatos N | Csak idő (bit) | Idő + összeg (bit) | Páronkénti csökkenés mediánja | Összeg, teljes megfigyelt idő (bit) |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows30:
        if r['model']=='reuse_mix':
            lines.append(f"| 1 → {r['k']} | {r['n_matched']} | {r['median_H_time_bits']:.3f} | {r['median_H_amount_bits']:.3f} | {r['median_delta_bits']:.3f} | {r['median_H_amount_full_bits']:.3f} |")
    lines += ['', 'A különbségek mediánja nem feltétlenül a két medián különbsége. A teljes megfigyelt idő oszlop ugyanazon 30 napnál találatos célokat tartja meg, de az összes megfigyelt későbbi kombinációt engedi. A végpont utáni ismeretlen történeteket ez sem tartalmazza.',
              '', '### Denominátorok és hiányzó találatok',
              '', '| Modell | Legalább 30 nap megfigyelés | Levágott | Nem levágott, nincs 30 napos találat | Egyedi a találatosakból |',
              '|---|---:|---:|---:|---:|']
    for r in rows30:
        if r['model']=='reuse_mix':lines.append(f"| 1 → {r['k']} | {r['n_targets']} | {r['n_capped']} | {r['n_no_match']} | {r['n_singleton']} |")
    lines += ['', 'Nulla találatnál a time-only entrópia továbbra is kiszámolható, az összegre feltételes posterior viszont nincs definiálva. Nem helyettesítjük nullával, és nem állítjuk, hogy a nulla találat bizonyítja a horizonton túli forrást.',
              '', '## Valószínűségi modell',
              '', 'Egy d befizetéshez, adott k mellett a time-only halmaz minden különböző k-kifizetéses részhalmaz a d utáni megfigyelt tranzakciókból. Ekkor még nincs összegszűrés, még a kifizetésenkénti összeghatár sem. A súly `a(U) = product_i f(t(w_i)-t(d))`; az összes súly normalizálása adja az időalapú eloszlást. Az összegváltozat ugyanezt az eloszlást kondicionálja a találatexportban szereplő kombinációkra. Minden eredmény adott k-ra feltételes; különböző kombinációméreteket nem keverünk prior nélkül.',
              '', 'A keverékkomponenst ebben az első számításban kifizetésenként függetlenül választjuk. Ez az egyszerűbb, független marginális késleltetésmodell, nem a korábban felvetett közös látens gyors/lassú állapot. A részleges kifizetések valós függőségét nem igazolja.',
              '', 'Az ablakon belül normalizált entrópia feltételes. Az `*_mass_inside_observed` oszlop az ablakba férő kombinációk tömegét adja a teljes megfigyelt időre normalizált modellben. Az `H_*_coarse_bits` megtartja a benti kombinációkat, és minden ablakon túli, de megfigyelt kombinációt egyetlen külső kategóriába von össze: ez információvesztő összesítés, nem teljes finanszírozási entrópia.',
              '', '## Időmodellek és érzékenység',
              '', f"A címújrahasználati minta N={fit['sample_n']}; átlag {fit['sample_mean_days']:.2f}, medián {fit['sample_median_days']:.2f} nap. A két exponenciális komponens átlaga {fit['means_days'][0]:.2f} és {fit['means_days'][1]:.2f} nap, súlya {fit['mixture_weights'][0]:.3f} és {fit['mixture_weights'][1]:.3f}. Ez kiválasztott párok leíró, cenzorálást nem korrigáló illesztése, nem a teljes PP-populáció kalibrált időeloszlása.",
              '', f"FIFO-forgatókönyv: az exponenciális átlagát a mentett FIFO-lag mediánjára ({fifo:.2f} nap) állítjuk. Little-forgatókönyv: {little:.2f} nap; az exportált átlagos egyenleg / átlagos kiáramlás aránya a {flows[0]['date']}–{flows[-1]['date']} közös intervallumon. A kettő érzékenységi feltevés, nem két további mért CDF.",
              '', '| Modell (30 nap) | k | Csak idő | Idő + összeg |', '|---|---:|---:|---:|']
    for r in rows30:lines.append(f"| {r['model']} | {r['k']} | {r['median_H_time_bits']:.3f} | {r['median_H_amount_bits']:.3f} |")
    lines += ['', '## Időhorizont és értelmezés',
              '', 'Az `entropy_comparison` ábra a teljes 180 napos követési idővel és már 3 napon belüli találattal rendelkező, nem levágott befizetéseket tartja fixen. A k=2 és k=3 minta nagyon kicsi; az ábra ezt feltünteti. Az `entropy_summary.csv` további bontásai: `available` (horizontonként teljes követés), `fixed_180d` (fix követési kohorsz, de horizontfüggő találatos részhalmaz), `common_matched` (a görbéken ténylegesen azonos összehasonlított célok).',
              '', 'Az összegfeltétel információt ad a választott modellen belül. A nagy bitkülönbség nem azonos helyes deanonymizációval: a time-only alap minden k-kombinációt megenged, a PP-kohorsz szűrt, és a levágott keresések kizárása szelektál. Különböző k-k és különböző kohorszok entrópiáiból nem szabad módszerrangsort képezni.',
              '', 'A cikk főszövegét ezzel az első számítással még nem írtuk felül. A CSV-k, a LaTeX táblázat és a vektoros ábrák a fenti korlátozásokkal emelhetők be. A modellfeltételeket és a bemenetek SHA-256 lenyomatait a `model_manifest.json` tartalmazza.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(cleaned(rows30),indent=2))
    print('Outputs:',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=PAPER.parent/'privacypools-deanonymization-main')
    parser.add_argument('--paper',type=Path,default=PAPER)
    parser.add_argument('--output',type=Path,default=PAPER/'analysis/pp_entropy')
    args=parser.parse_args()
    main(args.source.resolve(),args.output.resolve(),args.paper.resolve())
