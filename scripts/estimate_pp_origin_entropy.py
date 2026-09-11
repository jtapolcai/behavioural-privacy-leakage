#!/usr/bin/env python3
"""Same-support origin entropy; uncapped retrospective PP amount support.

Python orchestrates ingestion, time fitting, probabilities, tables and plots.
A compiled helper enumerates complete amount-group support without storing
combinatorially many witnesses. Model assumptions are exported with results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.special import gammaln
from estimate_pp_entropy import day, decimal_amount, entropy, fit_exp_mixture, log_density

PAPER = Path(__file__).resolve().parents[1]
SCALE = 10**12
TOL = 1000  # 1e-9 ETH, before any sum; Decimal ingestion avoids binary float parsing.


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def unique_rows(rows):
    return list({tuple(r.items()): r for r in rows}.values())


def load_events(source, dedup):
    base = source/'data/processed'
    dr = read(base/'processed_privacypools_eth_pool_deposits.csv')
    wr = read(base/'processed_privacypools_data_withdraws.csv')
    if dedup:
        dr, wr = unique_rows(dr), unique_rows(wr)
    deposits = sorted((day(r['time']), int((decimal_amount(r['amount_eth'])*SCALE).to_integral_value()))
                      for r in dr if decimal_amount(r['amount_eth']) > 0)
    agg = {}
    for r in wr:
        t = day(r['evt_block_time']); a = decimal_amount(r['eth_amount'])
        if a <= 0:
            raise ValueError('Nonpositive withdrawal amount')
        h = r['tx_hash']
        if h in agg:
            assert agg[h][0] == t
            agg[h][1] += a
        else:
            agg[h] = [t, a]
    end = min(deposits[-1][0], max(v[0] for v in agg.values()))
    withdrawals = sorted((t, int((a*SCALE).to_integral_value())) for t,a in agg.values() if t <= end)
    deposits = [d for d in deposits if d[0] <= end]
    return deposits, withdrawals, {'deposit_records': len(deposits), 'withdrawal_transactions':len(withdrawals),
        'withdrawals_excluded_after_shared_end':sum(v[0]>end for v in agg.values()),
        'shared_end_day':end, 'zero_deposit_rows_excluded':sum(decimal_amount(r['amount_eth'])<=0 for r in dr)}


def complete_support(deposits, withdrawals, executable, temp, horizon=None):
    inp, out = temp/'events.txt', temp/'support.bin'
    with inp.open('w') as f:
        f.write(f'{len(deposits)} {len(withdrawals)} {TOL}\n')
        for t,a in deposits+withdrawals:
            f.write(f'{t:.12f} {a}\n')
    args=[str(executable), str(inp), str(out)]
    if horizon is not None:
        args.append(str(horizon))
    subprocess.run(args, check=True)
    return np.fromfile(out, dtype=np.uint8).reshape(len(deposits),len(withdrawals))


def conditioned(p, support):
    q = p*support
    return q/q.sum() if q.sum()>0 else None


def robust(p, support, rho, feasible=None):
    """Sensitivity mixture; rho is assumed regime probability, not fitted confidence.

    No exact support means retention of the amount-feasible temporal baseline.
    The strict conditional entropy is separately recorded as undefined there.
    """
    background = p if feasible is None else conditioned(p, feasible)
    if background is None:
        raise ValueError('No amount-feasible origin under the single-origin model')
    q = conditioned(p, support)
    return background.copy() if q is None else (1-rho)*background+rho*q


def temporal_parameters(paper):
    reuse = np.array([float(r['dt_days']) for r in read(paper/'Figures/pp_h1_pairs.csv') if float(r['dt_days'])>0])
    fit = fit_exp_mixture(reuse)
    fifo = np.median([float(r['fifo_horizontal_lag_days']) for r in read(paper/'Figures/figure_ch4_03_pp_retained_fifo.csv') if r['fifo_horizontal_lag_days']])
    return reuse,fit,float(fifo)


def main(source, paper, output):
    output.mkdir(parents=True, exist_ok=True)
    reuse,fit,fifo = temporal_parameters(paper)
    models = ['reuse_mix','fifo_exp','little_exp','little_gamma_half','little_gamma_two']
    rows, horizons, metadata = [], [], {}
    with tempfile.TemporaryDirectory(prefix='pp-origin-') as tempname:
        temp = Path(tempname); executable = temp/'support'
        subprocess.run(['c++','-O3','-std=c++17',str(Path(__file__).with_name('pp_complete_support.cpp')),'-o',str(executable)],check=True)
        for scenario,dedup in [('exact_rows_collapsed',True),('as_exported',False)]:
            deposits,withdrawals,meta = load_events(source,dedup)
            dt = np.array([x[0] for x in deposits]); da = np.array([x[1] for x in deposits])
            wt = np.array([x[0] for x in withdrawals])
            # Recompute Little-style aggregate ratio from the same event scenario.
            start,end = dt.min(),meta['shared_end_day']; duration = end-start
            inflow_area = sum(a/SCALE*(end-t) for t,a in deposits)
            outflow_area = sum(a/SCALE*(end-t) for t,a in withdrawals)
            avg_balance = (inflow_area-outflow_area)/duration
            outflow = sum(a/SCALE for _,a in withdrawals)/duration
            little = avg_balance/outflow
            if little <= 0: raise ValueError('Nonpositive Little-style time scale')
            meta.update(little_mean_days=little,average_retained_eth=avg_balance,average_outflow_eth_per_day=outflow)
            print(f'{scenario}: complete amount support for {len(deposits)} deposits, {len(withdrawals)} withdrawals',flush=True)
            support = complete_support(deposits,withdrawals,executable,temp)
            assert not np.any(support[dt[:,None]>=wt[None,:]])
            meta['amount_support_pairs'] = int(np.count_nonzero(support))
            metadata[scenario] = meta
            for j,(t,amount) in enumerate(withdrawals):
                eligible = dt<t
                if not eligible.any():
                    raise ValueError('Withdrawal without preceding observed deposit; incomplete support')
                age=t-dt[eligible]; mask=support[eligible,j]>0; direct=support[eligible,j]==1
                for model in models:
                    if model.startswith('little_gamma'):
                        shape=.5 if model.endswith('half') else 2.
                        scale=little/shape
                        logs=(shape-1)*np.log(age)-age/scale-gammaln(shape)-shape*math.log(scale)
                    else:
                        logs=log_density(model,age,fit,fifo,little)
                    p=np.exp(logs-logs.max()); p/=p.sum()
                    q=conditioned(p,mask); q1=conditioned(p,direct)
                    feasible=da[eligible]+TOL>=amount
                    assert not np.any(mask & ~feasible)
                    r50=robust(p,mask,.5,feasible); r90=robust(p,mask,.9,feasible)
                    n=len(p); h=entropy(p)
                    assert h<=math.log2(n)+1e-10
                    row=dict(scenario=scenario,withdrawal_index=j,time_day=t,model=model,n_deposits=n,
                             followup_days=end-t,n_amount_origins=int(mask.sum()),n_direct_origins=int(direct.sum()),
                             H_uniform=math.log2(n),H_time=h,H_amount_feasible=entropy(conditioned(p,feasible)),H_knapsack_rho50=entropy(r50),H_knapsack_rho90=entropy(r90),
                             H_strict=entropy(q) if q is not None else math.nan,
                             H_direct_strict=entropy(q1) if q1 is not None else math.nan,
                             H0_amount=math.log2(mask.sum()) if mask.any() else math.nan,
                             time_mass_amount_support=float(p[mask].sum()),
                             time_mass_amount_infeasible=float(p[da[eligible]+TOL<amount].sum()),
                             max_p_time=float(p.max()),max_p_rho90=float(r90.max()),abstained=int(q is None))
                    rows.append(row)
                    for T in (3,7,30,86,180):
                        inside=age<=T
                        horizons.append(dict(scenario=scenario,withdrawal_index=j,model=model,T_days=T,
                            mass_time_inside=float(p[inside].sum()),mass_knapsack_rho50_inside=float(r50[inside].sum()),
                            H_time_inside_conditional=entropy(p[inside]),n_inside=int(inside.sum()),
                            reuse_sample_coverage=float(np.mean(reuse<=T))))
            print(f'{scenario}: entropy finished',flush=True)
    write(output/'per_withdrawal.csv',rows); write(output/'horizon_mass.csv',horizons)
    summary=[]
    metrics=['H_uniform','H_time','H_amount_feasible','H_knapsack_rho50','H_knapsack_rho90','H_strict','H_direct_strict','H0_amount']
    for scenario in metadata:
        for model in models:
            rr=[r for r in rows if r['scenario']==scenario and r['model']==model]
            for cohort in ('all','followup_90d'):
                cc=[r for r in rr if cohort=='all' or r['followup_days']>=90]
                for metric in metrics:
                    vals=np.array([r[metric] for r in cc]); valid=np.isfinite(vals)
                    summary.append(dict(scenario=scenario,model=model,cohort=cohort,metric=metric,n=len(cc),
                        n_defined=int(valid.sum()),n_abstained=sum(r['abstained'] for r in cc),
                        mean=float(np.mean(vals[valid])),median=float(np.median(vals[valid])),
                        p10=float(np.quantile(vals[valid],.1)),p90=float(np.quantile(vals[valid],.9))))
    write(output/'summary.csv',summary)
    horizon_summary=[]
    for scenario in metadata:
        for model in models:
            for T in (3,7,30,86,180):
                hh=[r for r in horizons if r['scenario']==scenario and r['model']==model and r['T_days']==T]
                horizon_summary.append(dict(scenario=scenario,model=model,T_days=T,n=len(hh),
                    mean_time_mass_inside=float(np.mean([r['mass_time_inside'] for r in hh])),
                    mean_rho50_mass_inside=float(np.mean([r['mass_knapsack_rho50_inside'] for r in hh])),
                    reuse_sample_coverage=hh[0]['reuse_sample_coverage']))
    write(output/'horizon_summary.csv',horizon_summary)
    assumptions={
        'support':'For each withdrawal, all strictly earlier positive deposit records through shared export coverage. Surrogate row identities, not people; deposit event IDs unavailable.',
        'deduplication':'Exact-row collapse and unmodified-export scenarios both reported. Neither establishes true event identity. Withdrawal rows summed per transaction hash after scenario handling.',
        'time_models':'Reuse mixture is selected-sample descriptive fit. FIFO median sets an assumed exponential mean. Little mean is recomputed from exported cumulative net flows with zero initial balance. Gamma shapes 0.5 and 2 test distributions with the same Little mean. No calibrated population holding-time distribution is available.',
        'time_prior':'p(d|w) proportional to f(tw-td), normalized over observed earlier deposits. A structural weighting assumption, not a posterior learned from ground-truth origin labels. No 30-day truncation.',
        'amount_support':'Complete existence of deposit = sum of 1, 2 or 3 distinct later withdrawal transactions through shared endpoint, including target w. No witness cap or address/roundness filter. Each origin counted once regardless of witness count. Gross eth_amount; tolerance 1e-9 ETH with fixed-point 1e-12 ETH ingestion.',
        'retrospective':'Other withdrawals may follow target withdrawal. Results are retrospective, not an online attacker estimate. Support omits >3-output decompositions, residual balances and later unseen withdrawals.',
        'mixture':'p3=(1-rho)*p_feasible + rho*p_time(. | amount support), where p_feasible conditions the temporal prior on deposit amount >= withdrawal amount (within tolerance). This necessary constraint assumes a single origin without subsequent balance increases. rho=0.5,0.9 are assumed conditional regime probabilities for sensitivity, NOT empirically fitted match correctness. Broad component retains feasible partial/incomplete explanations. NOT a certified entropy bound.',
        'zero_matches':'Strict conditional entropy undefined. Robust procedure abstains from exact-sum inference and retains amount-feasible temporal distribution. An empty feasible set raises an error instead of reporting false certainty.',
        'entropy':'Shannon entropy of stipulated origin distributions. H0_amount is Hartley support size, separately named. Neither is validated posterior privacy entropy. Conditioning may increase entropy for individual targets.',
        'missing_history':'No mass assigned to unobserved pre-export deposit events. Initial zero balance/complete start is an assumption to verify from chain data. Retaining all earlier OBSERVED deposits removes finite-window truncation, not unobserved-history bias.',
        'reuse_fit':fit,'fifo_mean_days':fifo,'scenarios':metadata}
    files=[Path(__file__),Path(__file__).with_name('pp_complete_support.cpp'),Path(__file__).with_name('estimate_pp_entropy.py'),
        source/'data/processed/processed_privacypools_eth_pool_deposits.csv',source/'data/processed/processed_privacypools_data_withdraws.csv',
        paper/'Figures/pp_h1_pairs.csv',paper/'Figures/figure_ch4_03_pp_retained_fifo.csv']
    assumptions['sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (output/'model_manifest.json').write_text(json.dumps(assumptions,indent=2)+'\n')
    render(output,rows,summary,assumptions)


def render(output, rows, summary, assumptions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rr=[r for r in rows if r['scenario']=='exact_rows_collapsed' and r['model']=='reuse_mix']
    fig,ax=plt.subplots(figsize=(7.4,4.2))
    for key,label in [('H_uniform','Uniform prior'),('H_time','Reuse-delay weighting'),('H_knapsack_rho50','Time + amount, rho=0.5'),('H_knapsack_rho90','Time + amount, rho=0.9')]:
        x=np.sort([r[key] for r in rr]); ax.plot(x,np.arange(1,len(x)+1)/len(x),label=label)
    x=np.sort([r['H_amount_feasible'] for r in rr])
    ax.plot(x,np.arange(1,len(x)+1)/len(x),'--',color='gray',label='Time + sufficient deposit amount')
    ax.set(xlabel='Model entropy of origin deposit (bits)',ylabel='Fraction of withdrawals',ylim=(0,1))
    ax.legend(fontsize=9); ax.grid(alpha=.2); fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(output/f'origin_entropy_cdf.{ext}',dpi=180)
    plt.close(fig)
    lines=['# PP: azonos eredetbefizetés-jelölteken számított entrópia','',
        'Teljes újrakeresés a nyers exportokból: nincs 200 találatos limit, cím- vagy kerekösszeg-szűrés. Egy kifizetéshez minden korábbi megfigyelt pozitív befizetés jelölt. A fő összevetésben a nulltalálatos kifizetések is benne maradnak.','',
        'Az alábbiak **feltételezett valószínűségi modellek entrópiái**, nem validált posterior privacy entropy. A rho az összegmodell alkalmazhatóságára felvett érzékenységi paraméter. Nincs adatból azonosított helyes értéke.','',
        '| Adatkezelés | Időmodell | n | egyenletes | idő | + knapsack, rho=.5 | + knapsack, rho=.9 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for scenario in assumptions['scenarios']:
        for model in ('reuse_mix','fifo_exp','little_exp','little_gamma_half','little_gamma_two'):
            ss={r['metric']:r for r in summary if r['scenario']==scenario and r['model']==model and r['cohort']=='all'}
            keys=['H_uniform','H_time','H_knapsack_rho50','H_knapsack_rho90']
            lines.append('| '+scenario+' | '+model+' | '+str(ss['H_time']['n'])+' | '+' | '.join(f"{ss[k]['median']:.3f}" for k in keys)+' |')
    lines += ['', 'A táblázat mediánokat mutat, bitekben. A summary.csv átlagokat és percentiliseket is tartalmaz, továbbá legalább 90 nap utómegfigyeléssel rendelkező kifizetésekre külön eredményt. A follow-up szerinti eltérés kohorszhatást is tartalmaz, nem tiszta cenzorálási korrekció.', '',
        f"Az összevont soros, reuse-időmodellel számított változatban {sum(r['n_amount_origins']>0 for r in rr)}/{len(rr)} kifizetésnek van egzakt összegtámogatása; {sum(r['n_amount_origins']==1 for r in rr)} esetben egyetlen támogatott befizetés marad. Ez nem igazolt azonosítás. A {sum(r['abstained'] for r in rr)} nulltalálatos esetet a kevert modellek nem hagyják ki.", '',
        'Az átlagos entrópia ugyanebben a változatban: '+', '.join(f"{label}: {np.mean([r[key] for r in rr]):.3f} bit" for key,label in [('H_time','idő'),('H_amount_feasible','idő + elégséges befizetésösszeg'),('H_knapsack_rho50','+ knapsack rho=.5'),('H_knapsack_rho90','+ knapsack rho=.9')])+'. Az egzakt felbontások többlethatását az elégséges befizetésösszegre már feltételes modellhez is viszonyítani kell.', '',
        '## Három modell', '',
        '1. p0(d|w)=1/N(w); H0_baseline=log2 N(w). Nincs 30 napos levágás.',
        '2. p_time(d|w) arányos f(tw-td)-vel, az összes korábbi megfigyelt befizetésen normalizálva.',
        '3. p_mix=(1-rho)p_feasible + rho p_time(.|S), ahol S a kifizetést tartalmazó teljes 1/2/3-kifizetéses felbontással rendelkező eredetbefizetések halmaza. p_feasible az időmodell a kifizetésnél legalább akkora befizetésekre feltételesen. Ez szükséges feltétel az egyetlen eredetbefizetést feltételező modellben. A sok felbontást adó befizetést nem sokszorozzuk meg. Ha S üres, megtartjuk p_feasible-t; a szigorú feltételes entrópia ilyenkor definiálatlan. A H_amount_feasible külön diagnosztika mutatja, mennyi hatás származik pusztán ebből az összegkorlátból.', '',
        '## Értelmezési korlátok', '',
        '- A knapsack retrospektív: a célkifizetés utáni, de az export közös végpontja előtti kifizetéseket is látja. Ez több megfigyelés, mint ami a célkifizetés pillanatában elérhető.',
        '- A rho nem találati helyesség mért becslése. A maradék komponens részleges felhasználást, háromnál több kifizetést és modellhibát enged meg. Nem tanultunk ezekre generatív modellt.',
        '- A reuse időeloszlás szelektált; ezt a rendelkezésre álló cím-újrahasználati párokból nem lehet torzítatlan populációs eloszlássá alakítani. FIFO és Little csak alternatív időskálát ad. Little átlagából önmagában nem következik exponenciális eloszlás: azonos átlagú gamma-változatokat is számolunk.',
        '- A deposit exportban nincs tranzakció-/eseményazonosító. Ezért az azonos sorok összevonása érzékenységi forgatókönyv, nem bizonyított duplikációjavítás. A kifizetéseket hash szerint összegezzük.',
        '- Az összegeknél a teljes eth_amount szerepel, nem a relayer díja utáni nettó kifizetés. Az összegegyezés toleranciája 1e-9 ETH; a nulla értékű deposit sorok nem finanszírozhatnak pozitív kifizetést.',
        '- Az induló egyenleget nullának feltételezzük; a láncon ellenőrzött eseményteljesség nélkül ez nem bizonyított teljes történet. Személyekre vonatkozó entrópia nem azonos a befizetésrekordok entrópiájával.',
        '- Az egyes kifizetések eredetei külön hipotézisek; nem oldjuk meg a minden kifizetésre közös, dupla felhasználást kizáró globális történet posteriorát.',
        '- A horizon_mass.csv teljes eloszlások ablakon belüli tömegét méri. Az azon kívüli befizetések az entrópiában külön jelöltek maradnak. F_reuse(T) kizárólag a reuse minta lefedettsége.',
        '- Szigorú feltételre az egyedi entrópia akár nőhet is: az ilyen eseteket nem vágjuk nullára.','',
        'Az előző analysis/pp_entropy eredmény más véletlen változóról, rögzített méretű kifizetéskombinációkról szólt. Az itt közölt eredetentrópiával nem közvetlenül összehasonlítható.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    tex=['% Generated by estimate_pp_origin_entropy.py; medians; rho is assumed.',r'\begin{tabular}{lrrrr}',r'\hline',r'Time model & Uniform & Time & $\rho=.5$ & $\rho=.9$ \\',r'\hline']
    for model,label in [('reuse_mix','Reuse mixture'),('fifo_exp','FIFO exponential'),('little_exp','Little exponential'),('little_gamma_half',r'Little $\Gamma(0.5)$'),('little_gamma_two',r'Little $\Gamma(2)$')]:
        ss={r['metric']:r for r in summary if r['scenario']=='exact_rows_collapsed' and r['model']==model and r['cohort']=='all'}
        tex.append(label+' & '+' & '.join(f"{ss[k]['median']:.2f}" for k in ['H_uniform','H_time','H_knapsack_rho50','H_knapsack_rho90'])+r' \\')
    tex += [r'\hline',r'\end{tabular}']
    (output/'origin_entropy_table.tex').write_text('\n'.join(tex)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,default=PAPER.parent/'privacypools-deanonymization-main')
    ap.add_argument('--paper',type=Path,default=PAPER); ap.add_argument('--output',type=Path,default=PAPER/'analysis/pp_origin_entropy')
    args=ap.parse_args(); main(args.source.resolve(),args.paper.resolve(),args.output.resolve())
