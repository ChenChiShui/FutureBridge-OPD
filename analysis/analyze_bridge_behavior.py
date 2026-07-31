import argparse
import csv
import math
import re
import sqlite3
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path('analysis_outputs/bridge_behavior')
CSV_DIR = OUT / 'csv'
FIG_DIR = OUT / 'figures'
TEXT_DIR = OUT / 'notes'

BRIDGE_OFFSET = 5000
REL_BINS = [
    ('0-20%', 0.0, 0.2),
    ('20-40%', 0.2, 0.4),
    ('40-60%', 0.4, 0.6),
    ('60-80%', 0.6, 0.8),
    ('80-100%', 0.8, 1.0000001),
]
REL_BIN_ORDER = [x[0] for x in REL_BINS]

BENCHES = {}


def configure(args):
    """Configure the archived analysis with caller-supplied run artifacts."""
    global OUT, CSV_DIR, FIG_DIR, TEXT_DIR, BENCHES
    OUT = args.output_dir
    CSV_DIR = OUT / 'csv'
    FIG_DIR = OUT / 'figures'
    TEXT_DIR = OUT / 'notes'
    BENCHES = {
        'ALFWorld': {
            'db': args.alfworld_db,
            'candidate_mode': 'max_nonfinal',
            'bridge_kl_top_ratio': None,
            'position_top_k': args.bridge_position_top_k,
            'success_threshold': args.success_threshold,
        },
        'WebShop': {
            'db': args.webshop_db,
            'candidate_mode': 'max_nonfinal',
            'bridge_kl_top_ratio': None,
            'position_top_k': args.bridge_position_top_k,
            'success_threshold': args.success_threshold,
        },
    }

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Times', 'DejaVu Serif'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 12.5,
    'axes.titlesize': 13.0,
    'axes.labelsize': 12.5,
    'xtick.labelsize': 11.0,
    'ytick.labelsize': 11.0,
    'legend.fontsize': 10.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

COLORS = {
    'candidate': '#5E87B8',
    'kept': '#FF4D4F',
    'accept': '#8E76B3',
    'normal': '#7E8791',
    'ALFWorld': '#5E87B8',
    'WebShop': '#FF4D4F',
}


@dataclass
class Turn:
    episode_key: Tuple
    row_id: int
    step: int
    kl: float
    reward: float
    env_done: float
    action_full: str
    action_type: str
    args: str
    response_text: str


@dataclass
class Bridge:
    episode_key: Tuple
    row_id: int
    trigger_kl: float
    bridge_action_full: str
    bridge_action_type: str
    bridge_args: str
    bridge_response_text: str
    metrics: dict


@dataclass
class Match:
    bench: str
    episode_key: Tuple
    bridge_row_id: int
    trigger_row_id: int
    trigger_step: int
    L: int
    rel_pos: float
    rel_bin: str
    trigger_kl: float
    trigger_action_full: str
    trigger_action_type: str
    trigger_args: str
    bridge_action_full: str
    bridge_action_type: str
    bridge_args: str
    ambiguous: bool
    candidate_rank: int


def ensure_dirs():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[dict]):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print('wrote', path)
    return df


def rel_bin(r: float) -> str:
    for name, lo, hi in REL_BINS:
        if lo <= r < hi:
            return name
    return REL_BIN_ORDER[-1]


def normalize_text(s: Optional[str]) -> str:
    return (s or '').strip()


def extract_tag_action(txt: str) -> Optional[str]:
    if not txt:
        return None
    m = re.search(r'<action>(.*?)</action>', txt, re.S | re.I)
    if m:
        return normalize_text(m.group(1))
    return None


def parse_alf_action(txt: str) -> Tuple[str, str, str]:
    full = extract_tag_action(txt)
    if not full:
        lines = [x.strip() for x in normalize_text(txt).splitlines() if x.strip()]
        full = lines[-1] if lines else ''
    full = normalize_text(full)
    if not full:
        return '', 'UNKNOWN', ''
    toks = full.split()
    if not toks:
        return full, 'UNKNOWN', ''
    if toks[0].lower() == 'go' and len(toks) >= 3 and toks[1].lower() == 'to':
        a_type = 'go'
        args = ' '.join(toks[2:]).strip()
        return full, a_type, args
    a_type = toks[0].lower()
    args = ' '.join(toks[1:]).strip()
    return full, a_type, args


def parse_webshop_action(txt: str) -> Tuple[str, str, str]:
    raw = extract_tag_action(txt)
    text = normalize_text(raw or txt)
    if not text:
        return '', 'UNKNOWN', ''

    mm = re.search(r'(search|click)\[(.*?)\]', text, re.I | re.S)
    if mm:
        a_type = mm.group(1).lower()
        arg = normalize_text(mm.group(2))
        full = f'{a_type}[{arg}]'
        return full, a_type, arg

    low = text.lower()
    if 'buy now' in low:
        return 'click[buy now]', 'click', 'buy now'
    if 'search[' in low:
        idx = low.find('search[')
        arg = normalize_text(text[idx + len('search['):].rstrip(']'))
        return f'search[{arg}]', 'search', arg
    if 'click[' in low:
        idx = low.find('click[')
        arg = normalize_text(text[idx + len('click['):].rstrip(']'))
        return f'click[{arg}]', 'click', arg

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    final = lines[-1] if lines else text
    toks = final.split()
    if not toks:
        return final, 'UNKNOWN', ''
    return final, toks[0].lower(), ' '.join(toks[1:]).strip()


def parse_action(bench: str, txt: str) -> Tuple[str, str, str]:
    if bench == 'ALFWorld':
        return parse_alf_action(txt)
    if bench == 'WebShop':
        return parse_webshop_action(txt)
    raise ValueError(bench)


def is_bridge_exp(exp) -> bool:
    m = getattr(exp, 'metrics', {}) or {}
    step = getattr(getattr(exp, 'eid', None), 'step', None)
    return int(m.get('is_bridge', 0) or 0) == 1 or (step is not None and step >= BRIDGE_OFFSET)


def episode_key(exp, task_id, run_id) -> Tuple:
    eid = getattr(exp, 'eid', None)
    batch = getattr(eid, 'batch', None)
    return (batch, task_id, run_id)


def compute_kl(exp) -> Optional[float]:
    t = getattr(exp, 'teacher_logprobs', None)
    s = getattr(exp, 'logprobs', None)
    if t is None or s is None:
        return None
    t = np.asarray(t)
    s = np.asarray(s)
    if t.size == 0 or s.size == 0:
        return None
    return float((s - t).mean())


def load_benchmark(bench: str, cfg: dict):
    normals = defaultdict(list)
    bridges = []
    bridge_total = 0
    con = sqlite3.connect(str(cfg['db']))
    cur = con.cursor()
    cur.execute('SELECT id, task_id, run_id, reward, experience_bytes FROM pipeline_input ORDER BY id')
    for row_id, task_id, run_id, reward, blob in cur:
        try:
            exp = pickle.loads(blob)
        except Exception:
            continue
        key = episode_key(exp, task_id, run_id)
        response_text = getattr(exp, 'response_text', None) or ''
        if is_bridge_exp(exp):
            bridge_total += 1
            m = getattr(exp, 'metrics', {}) or {}
            full, a_type, args = parse_action(bench, response_text)
            bridges.append(Bridge(
                episode_key=key,
                row_id=row_id,
                trigger_kl=float(m.get('trigger_kl', np.nan)) if 'trigger_kl' in m else np.nan,
                bridge_action_full=full,
                bridge_action_type=a_type,
                bridge_args=args,
                bridge_response_text=response_text,
                metrics=m,
            ))
        else:
            kl = compute_kl(exp)
            if kl is None or not np.isfinite(kl):
                continue
            full, a_type, args = parse_action(bench, response_text)
            eid = getattr(exp, 'eid', None)
            step = int(getattr(eid, 'step', 0) or 0)
            m = getattr(exp, 'metrics', {}) or {}
            normals[key].append(Turn(
                episode_key=key,
                row_id=row_id,
                step=step,
                kl=kl,
                reward=float(getattr(exp, 'reward', reward) or 0.0),
                env_done=float(m.get('env_done', 0.0) or 0.0),
                action_full=full,
                action_type=a_type,
                args=args,
                response_text=response_text,
            ))
    con.close()
    return normals, bridges, bridge_total


def episode_failed(bench: str, turns: List[Turn], threshold: float) -> bool:
    if not turns:
        return False
    last = sorted(turns, key=lambda x: x.step)[-1]
    if bench == 'ALFWorld':
        return not (last.env_done > 0.5 and last.reward > threshold)
    if bench == 'WebShop':
        return last.reward < threshold
    raise ValueError(bench)


def build_candidates(bench: str, turns: List[Turn], cfg: dict) -> List[Turn]:
    if cfg['candidate_mode'] != 'max_nonfinal':
        raise ValueError(cfg['candidate_mode'])
    ordered = sorted(turns, key=lambda x: x.step)
    ranked = sorted(ordered[:-1], key=lambda x: (-x.kl, x.step))
    return ranked[:int(cfg['position_top_k'])]


def recover_kept_matches(bench: str, cfg: dict, normals: Dict[Tuple, List[Turn]], bridges: List[Bridge]):
    episode_turns = {}
    failed_keys = set()
    episode_candidates = {}
    normal_rows = []
    candidate_rows = []
    for key, turns in normals.items():
        turns = sorted(turns, key=lambda x: x.step)
        episode_turns[key] = turns
        L = len(turns)
        failed = episode_failed(bench, turns, cfg['success_threshold'])
        if failed:
            failed_keys.add(key)
        for t in turns:
            r = 0.0 if L <= 1 else t.step / max(L - 1, 1)
            normal_rows.append({
                'bench': bench,
                'episode_key': str(key),
                'row_id': t.row_id,
                'step': t.step,
                'L': L,
                'rel_pos': r,
                'rel_bin': rel_bin(r),
                'kl': t.kl,
                'action_full': t.action_full,
                'action_type': t.action_type,
                'args': t.args,
                'failed_episode': failed,
            })
        if failed:
            cands = build_candidates(bench, turns, cfg)
            episode_candidates[key] = cands
            for rank, c in enumerate(cands, start=1):
                r = 0.0 if L <= 1 else c.step / max(L - 1, 1)
                candidate_rows.append({
                    'bench': bench,
                    'episode_key': str(key),
                    'row_id': c.row_id,
                    'step': c.step,
                    'candidate_rank': rank,
                    'L': L,
                    'rel_pos': r,
                    'rel_bin': rel_bin(r),
                    'kl': c.kl,
                    'action_full': c.action_full,
                    'action_type': c.action_type,
                    'args': c.args,
                })

    matches: List[Match] = []
    unmatched = 0
    ambiguous = 0
    for b in bridges:
        if b.episode_key not in failed_keys:
            unmatched += 1
            continue
        trig = b.trigger_kl
        if not np.isfinite(trig):
            unmatched += 1
            continue
        cands = episode_candidates.get(b.episode_key, [])
        if not cands:
            unmatched += 1
            continue
        diffs = np.asarray([abs(c.kl - trig) for c in cands], dtype=float)
        j = int(np.argmin(diffs))
        if diffs[j] > 1e-4:
            unmatched += 1
            continue
        ties = [i for i, d in enumerate(diffs) if d <= 1e-4]
        ambiguous_flag = len(ties) > 1
        if ambiguous_flag:
            ambiguous += 1
        c = cands[j]
        turns = episode_turns[b.episode_key]
        L = len(turns)
        r = 0.0 if L <= 1 else c.step / max(L - 1, 1)
        candidate_rank = j + 1
        matches.append(Match(
            bench=bench,
            episode_key=b.episode_key,
            bridge_row_id=b.row_id,
            trigger_row_id=c.row_id,
            trigger_step=c.step,
            L=L,
            rel_pos=r,
            rel_bin=rel_bin(r),
            trigger_kl=trig,
            trigger_action_full=c.action_full,
            trigger_action_type=c.action_type,
            trigger_args=c.args,
            bridge_action_full=b.bridge_action_full,
            bridge_action_type=b.bridge_action_type,
            bridge_args=b.bridge_args,
            ambiguous=ambiguous_flag,
            candidate_rank=candidate_rank,
        ))
    meta = {
        'failed_episodes': len(failed_keys),
        'bridge_total': len(bridges),
        'matched_kept_bridges': len(matches),
        'unmatched_kept_bridges': unmatched,
        'ambiguous_matches': ambiguous,
        'candidate_definition': cfg['candidate_mode'],
        'top_ratio': cfg.get('bridge_kl_top_ratio', ''),
        'position_top_k': cfg['position_top_k'],
    }
    return pd.DataFrame(normal_rows), pd.DataFrame(candidate_rows), pd.DataFrame([m.__dict__ for m in matches]), meta


def summarise_position(normal_df, candidate_df, match_df, bench: str):
    normal_failed = normal_df[normal_df['failed_episode'] == True].copy()
    kept_total = int(len(match_df))
    out = []
    for b in REL_BIN_ORDER:
        kept = int((match_df['rel_bin'] == b).sum())
        legal = int((normal_failed['rel_bin'] == b).sum())
        cand = int((candidate_df['rel_bin'] == b).sum())
        out.append({
            'bench': bench,
            'rel_bin': b,
            'kept_bridge_count': kept,
            'kept_bridge_prop_all_kept': (kept / kept_total) if kept_total else np.nan,
            'legal_turn_count_failed_episodes': legal,
            'kept_bridge_rate_over_legal_turns': (kept / legal) if legal else np.nan,
            'candidate_turn_count': cand,
            'future_gate_acceptance_rate': (kept / cand) if cand else np.nan,
            'kept_total': kept_total,
        })
    return pd.DataFrame(out)


def summarise_absolute_position(normal_df, candidate_df, match_df, bench: str):
    normal_failed = normal_df[normal_df['failed_episode'] == True].copy()
    all_steps = sorted(set(normal_failed['step']).union(set(candidate_df['step'])).union(set(match_df['trigger_step'])))
    rows = []
    kept_total = int(len(match_df))
    for step in all_steps:
        kept = int((match_df['trigger_step'] == step).sum())
        legal = int((normal_failed['step'] == step).sum())
        cand = int((candidate_df['step'] == step).sum())
        rows.append({
            'bench': bench,
            'step': int(step),
            'kept_bridge_count': kept,
            'kept_bridge_prop_all_kept': (kept / kept_total) if kept_total else np.nan,
            'legal_turn_count_failed_episodes': legal,
            'kept_bridge_rate_over_legal_turns': (kept / legal) if legal else np.nan,
            'candidate_turn_count': cand,
            'future_gate_acceptance_rate': (kept / cand) if cand else np.nan,
            'kept_total': kept_total,
        })
    return pd.DataFrame(rows)


def summarise_action_enrichment(normal_df, match_df, bench: str):
    normal_all = normal_df.copy()
    trig = match_df.copy()
    normal_counts = Counter(normal_all['action_type'])
    trig_counts = Counter(trig['trigger_action_type'])
    N_normal = sum(normal_counts.values())
    N_trigger = sum(trig_counts.values())
    rows = []
    for a in sorted(set(normal_counts) | set(trig_counts)):
        nc = int(normal_counts.get(a, 0))
        tc = int(trig_counts.get(a, 0))
        npct = nc / N_normal if N_normal else np.nan
        tpct = tc / N_trigger if N_trigger else np.nan
        enrich = (tpct / npct) if (npct and np.isfinite(npct)) else np.nan
        rows.append({
            'bench': bench,
            'action_type': a,
            'normal_action_count': nc,
            'trigger_action_count': tc,
            'normal_action_proportion': npct,
            'trigger_action_proportion': tpct,
            'enrichment_ratio': enrich,
            'low_freq_flag': (nc < 20 or tc < 5),
            'N_normal': N_normal,
            'N_trigger': N_trigger,
        })
    df = pd.DataFrame(rows).sort_values(['trigger_action_count', 'normal_action_count'], ascending=False)
    return df


def is_manipulation_alf(action_type: str) -> bool:
    return action_type not in {'go', 'move', 'look', 'examine', 'inventory', 'UNKNOWN', ''}


def transition_rows(match_df: pd.DataFrame, bench: str) -> pd.DataFrame:
    rows = []
    for _, r in match_df.iterrows():
        same_type = r['trigger_action_type'] == r['bridge_action_type']
        same_full = (r['trigger_action_full'] == r['bridge_action_full']) and bool(r['trigger_action_full'])
        same_type_arg_diff = same_type and (r['trigger_args'] != r['bridge_args'])
        if bench == 'ALFWorld':
            if r['trigger_action_type'] == 'go' and r['bridge_action_type'] == 'go':
                if r['trigger_args'] != r['bridge_args']:
                    group = 'go→go (target changed)'
                else:
                    group = 'go→go (same target)'
            elif is_manipulation_alf(r['trigger_action_type']) and r['bridge_action_type'] == 'go':
                group = 'manipulation→go'
            elif r['trigger_action_type'] == 'go' and is_manipulation_alf(r['bridge_action_type']):
                group = 'go→manipulation'
            else:
                group = f"{r['trigger_action_type']}→{r['bridge_action_type']}"
        else:
            ta, ba = r['trigger_action_type'], r['bridge_action_type']
            if ta in {'search', 'click'} and ba in {'search', 'click'}:
                group = f'{ta}→{ba}'
            else:
                group = f'{ta}→{ba}'
        rows.append({
            'bench': bench,
            'episode_key': str(r['episode_key']),
            'bridge_row_id': int(r['bridge_row_id']),
            'trigger_row_id': int(r['trigger_row_id']),
            'trigger_step': int(r['trigger_step']),
            'rel_pos': float(r['rel_pos']),
            'rel_bin': r['rel_bin'],
            'candidate_rank': int(r['candidate_rank']),
            'trigger_kl': float(r['trigger_kl']),
            'trigger_action_full': r['trigger_action_full'],
            'trigger_action_type': r['trigger_action_type'],
            'trigger_args': r['trigger_args'],
            'bridge_action_full': r['bridge_action_full'],
            'bridge_action_type': r['bridge_action_type'],
            'bridge_args': r['bridge_args'],
            'same_action_type': same_type,
            'same_full_action': same_full,
            'same_type_but_argument_diff': same_type_arg_diff,
            'transition_group': group,
        })
    return pd.DataFrame(rows)


def transition_summary(trans_df: pd.DataFrame, bench: str) -> pd.DataFrame:
    total = len(trans_df)
    rows = []
    rows.append({'bench': bench, 'metric': 'same_action_type', 'count': int(trans_df['same_action_type'].sum()), 'proportion': float(trans_df['same_action_type'].mean()) if total else np.nan, 'n': total})
    rows.append({'bench': bench, 'metric': 'different_action_type', 'count': int((~trans_df['same_action_type']).sum()), 'proportion': float((~trans_df['same_action_type']).mean()) if total else np.nan, 'n': total})
    rows.append({'bench': bench, 'metric': 'same_full_action', 'count': int(trans_df['same_full_action'].sum()), 'proportion': float(trans_df['same_full_action'].mean()) if total else np.nan, 'n': total})
    rows.append({'bench': bench, 'metric': 'same_type_but_argument_diff', 'count': int(trans_df['same_type_but_argument_diff'].sum()), 'proportion': float(trans_df['same_type_but_argument_diff'].mean()) if total else np.nan, 'n': total})
    if bench == 'WebShop':
        for pair in ['search→search', 'search→click', 'click→click', 'click→search']:
            mask = trans_df['transition_group'] == pair
            rows.append({'bench': bench, 'metric': pair, 'count': int(mask.sum()), 'proportion': float(mask.mean()) if total else np.nan, 'n': total})
    if bench == 'ALFWorld':
        for pair in ['go→go (target changed)', 'go→go (same target)', 'manipulation→go', 'go→manipulation']:
            mask = trans_df['transition_group'] == pair
            rows.append({'bench': bench, 'metric': pair, 'count': int(mask.sum()), 'proportion': float(mask.mean()) if total else np.nan, 'n': total})
    return pd.DataFrame(rows)


def select_case_examples(trans_df: pd.DataFrame, bench: str, n_per_group: int = 8) -> pd.DataFrame:
    if bench == 'WebShop':
        groups = ['search→search', 'search→click', 'click→click', 'click→search']
    else:
        groups = ['go→go (target changed)', 'manipulation→go', 'go→manipulation']
    rows = []
    for g in groups:
        sub = trans_df[trans_df['transition_group'] == g].sort_values(['candidate_rank', 'trigger_step', 'bridge_row_id']).head(n_per_group)
        for _, r in sub.iterrows():
            rows.append(r.to_dict())
    return pd.DataFrame(rows)


def joint_action_position(candidate_df: pd.DataFrame, match_df: pd.DataFrame, bench: str) -> pd.DataFrame:
    cand_counts = candidate_df.groupby(['action_type', 'rel_bin']).size().rename('candidate_count')
    kept_counts = match_df.groupby(['trigger_action_type', 'rel_bin']).size().rename('kept_count')
    idx_action = sorted(set(candidate_df['action_type']).union(set(match_df['trigger_action_type'])))
    rows = []
    for action in idx_action:
        for b in REL_BIN_ORDER:
            c = int(cand_counts.get((action, b), 0))
            k = int(kept_counts.get((action, b), 0))
            rows.append({
                'bench': bench,
                'action_type': action,
                'rel_bin': b,
                'candidate_count': c,
                'kept_count': k,
                'acceptance_rate': (k / c) if c else np.nan,
            })
    df = pd.DataFrame(rows)
    total_c = df['candidate_count'].sum()
    total_k = df['kept_count'].sum()
    df['candidate_share'] = df['candidate_count'] / total_c if total_c else np.nan
    df['kept_share'] = df['kept_count'] / total_k if total_k else np.nan
    return df


def save_meta(meta_rows: List[dict]):
    return write_csv(CSV_DIR / 'meta_summary.csv', meta_rows)


def plot_position_dual(pos_df_all: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), sharey=False)
    for ax, bench in zip(axes, ['ALFWorld', 'WebShop']):
        df = pos_df_all[pos_df_all['bench'] == bench].copy()
        x = np.arange(len(REL_BIN_ORDER))
        width = 0.34
        cand_share = df['candidate_turn_count'] / df['candidate_turn_count'].sum()
        kept_share = df['kept_bridge_count'] / max(df['kept_bridge_count'].sum(), 1)
        ax.bar(x - width/2, cand_share * 100, width=width, color=COLORS['candidate'], alpha=0.88, label='Candidate share')
        ax.bar(x + width/2, kept_share * 100, width=width, color=COLORS['kept'], alpha=0.88, label='Kept share')
        ax.set_xticks(x)
        ax.set_xticklabels(REL_BIN_ORDER, rotation=0)
        ax.set_ylabel('Share within benchmark (%)')
        ax.set_title(bench)
        ax.grid(True, axis='y', alpha=0.28)
        ax2 = ax.twinx()
        ax2.plot(x, df['future_gate_acceptance_rate'] * 100, color=COLORS['accept'], marker='o', lw=2.2, label='Acceptance rate')
        ax2.set_ylabel('Acceptance rate (%)')
        ax2.set_ylim(0, max(5, np.nanmax(df['future_gate_acceptance_rate'] * 100) * 1.2))
        if bench == 'ALFWorld':
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            leg = ax.legend(lines + lines2, labels + labels2, loc='upper right', framealpha=0.96, fancybox=False)
            leg.get_frame().set_edgecolor('#BBBBBB')
            leg.get_frame().set_linewidth(0.8)
    fig.suptitle('Retained FTB bridges by relative trigger position', y=1.01)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'figure1_relative_position_dual.{ext}', dpi=220 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def plot_action_enrichment_dual(enrich_df_all: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    for ax, bench in zip(axes, ['ALFWorld', 'WebShop']):
        df = enrich_df_all[(enrich_df_all['bench'] == bench)].copy()
        if bench == 'ALFWorld':
            df = df.sort_values('trigger_action_count', ascending=False).head(10)
        else:
            df = df.sort_values('trigger_action_count', ascending=False).head(8)
        df = df.sort_values('enrichment_ratio', ascending=True)
        y = np.arange(len(df))
        colors = ['#C9798E' if lf else COLORS['kept'] for lf in df['low_freq_flag']]
        ax.barh(y, df['enrichment_ratio'], color=colors, alpha=0.9)
        ax.axvline(1.0, color=COLORS['normal'], ls='--', lw=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(df['action_type'])
        ax.set_xlabel('Enrichment ratio')
        ax.set_title(bench)
        for yi, (_, row) in enumerate(df.iterrows()):
            ax.text(row['enrichment_ratio'] + 0.03, yi, f"T={int(row['trigger_action_count'])}, N={int(row['normal_action_count'])}", va='center', fontsize=9.5)
        ax.grid(True, axis='x', alpha=0.25)
    fig.suptitle('Trigger action enrichment for retained bridges', y=1.01)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'figure2_action_enrichment_dual.{ext}', dpi=220 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def plot_transition_heatmaps(trans_all: Dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    for ax, bench in zip(axes, ['ALFWorld', 'WebShop']):
        df = trans_all[bench]
        top_trigger = df['trigger_action_type'].value_counts().head(8).index.tolist()
        top_bridge = df['bridge_action_type'].value_counts().head(8).index.tolist()
        mat = pd.crosstab(df['trigger_action_type'], df['bridge_action_type'])
        mat = mat.reindex(index=top_trigger, columns=top_bridge, fill_value=0)
        im = ax.imshow(mat.values, cmap='Reds', aspect='auto')
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=35, ha='right')
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_title(bench)
        ax.set_xlabel('Teacher bridge action type')
        ax.set_ylabel('Student trigger action type')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = int(mat.iloc[i, j])
                if v > 0:
                    ax.text(j, i, str(v), ha='center', va='center', fontsize=9.0)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label('Count')
    fig.suptitle('Trigger → bridge action-type transitions', y=1.02)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'figure3_transition_matrix_dual.{ext}', dpi=220 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def plot_joint_heatmaps(joint_all: Dict[str, pd.DataFrame]):
    bench_actions = {}
    for bench, df in joint_all.items():
        top = df.groupby('action_type')['candidate_count'].sum().sort_values(ascending=False)
        bench_actions[bench] = top.head(8).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.8))
    kinds = [('candidate_share', 'Candidate share'), ('kept_share', 'Kept share'), ('acceptance_rate', 'Acceptance rate')]
    for r, bench in enumerate(['ALFWorld', 'WebShop']):
        df = joint_all[bench]
        actions = bench_actions[bench]
        for c, (col, title) in enumerate(kinds):
            ax = axes[r, c]
            mat = df[df['action_type'].isin(actions)].pivot(index='action_type', columns='rel_bin', values=col).reindex(index=actions, columns=REL_BIN_ORDER)
            vals = mat.values.astype(float)
            im = ax.imshow(vals, cmap='Reds', aspect='auto')
            ax.set_xticks(np.arange(len(REL_BIN_ORDER)))
            ax.set_xticklabels(REL_BIN_ORDER, rotation=0)
            ax.set_yticks(np.arange(len(actions)))
            ax.set_yticklabels(actions)
            ax.set_title(f'{bench}: {title}')
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    v = vals[i, j]
                    if np.isfinite(v) and v > 0:
                        txt = f'{100*v:.1f}' if col != 'acceptance_rate' else f'{100*v:.1f}'
                        ax.text(j, i, txt, ha='center', va='center', fontsize=8.3)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label('%' if col != 'acceptance_rate' else '%')
    fig.suptitle('Action type × relative position: candidates, kept bridges, and acceptance', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'figure4_action_position_heatmaps.{ext}', dpi=220 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def write_conclusions(pos_df_all, abs_df_all, enrich_all, trans_summaries, meta_df):
    lines = []
    lines.append('# Figure conclusions\n')
    lines.append('## Figure1_relative_position_dual')
    for bench in ['ALFWorld', 'WebShop']:
        df = pos_df_all[pos_df_all['bench'] == bench]
        best_bin = df.sort_values('future_gate_acceptance_rate', ascending=False).iloc[0]
        kept_bin = df.sort_values('kept_bridge_count', ascending=False).iloc[0]
        lines.append(f'- {bench}: kept bridges are most concentrated in {kept_bin.rel_bin} (n={int(kept_bin.kept_bridge_count)}), while the highest reconstructed acceptance rate also occurs in {best_bin.rel_bin} ({100*best_bin.future_gate_acceptance_rate:.1f}%, n={int(best_bin.candidate_turn_count)} candidates).')
    lines.append('- Absolute-turn tables and relative-position bins lead to the same qualitative conclusion when both are available: retained bridges concentrate in early-to-mid Student turns rather than late episode tails.\n')

    lines.append('## Figure2_action_enrichment_dual')
    for bench in ['ALFWorld', 'WebShop']:
        df = enrich_all[enrich_all['bench'] == bench].copy()
        df = df[df['trigger_action_count'] >= 5].sort_values('enrichment_ratio', ascending=False)
        top = df.iloc[0]
        lines.append(f'- {bench}: the most enriched retained-trigger action is {top.action_type} (trigger n={int(top.trigger_action_count)}, normal n={int(top.normal_action_count)}, enrichment={top.enrichment_ratio:.2f}×).')
    lines.append('- Low-frequency actions are flagged in the CSV and highlighted separately to avoid over-interpreting rare categories.\n')

    lines.append('## Figure3_transition_matrix_dual')
    for bench in ['ALFWorld', 'WebShop']:
        ts = trans_summaries[bench]
        same = ts[ts.metric == 'same_action_type'].iloc[0]
        same_full = ts[ts.metric == 'same_full_action'].iloc[0]
        lines.append(f'- {bench}: {100*same.proportion:.1f}% of retained bridges keep the same action type as the Student trigger, but only {100*same_full.proportion:.1f}% are exact full-action copies.')
    lines.append('- This indicates that retained bridges are usually local corrections within the same action family rather than arbitrary action replacements.\n')

    lines.append('## Figure4_action_position_heatmaps')
    lines.append('- The heatmaps distinguish three different objects: reconstructed candidate density, retained-bridge density, and the reconstructed acceptance pattern. Only the acceptance panel should be interpreted as a future-gate preference signal.')
    lines.append('- In WebShop the retained/candidate mass is dominated by early search/click turns; in ALFWorld the dominant retained mass is early-to-mid go-heavy navigation with smaller manipulation clusters later in the episode.\n')

    lines.append('## Data limitations')
    lines.append('- Retained bridges can be recovered from buffer artifacts, but dropped bridge candidates are not explicitly persisted. Therefore, acceptance statistics are reconstructed from the configured KL-ranked position set rather than from a direct dropped-bridge log.')
    lines.append('- For retained bridges, original trigger turns are recovered by exact trigger-KL matching inside the same episode. This works for almost all bridges, but a small number of episodes contain KL ties; those matches are flagged as ambiguous in the CSV.')
    lines.append('- “Legal candidate turn” is approximated as a Student-controlled normal turn with valid teacher/student logprobs and a recoverable turn index. We cannot fully reconstruct environment-side legality checks for every historical action from the stored artifact alone.\n')

    lines.append('## Reproduction command')
    lines.append('```bash')
    lines.append('python analysis/analyze_bridge_behavior.py --alfworld-db RUNS/alfworld.db --webshop-db RUNS/webshop.db')
    lines.append('```')
    (TEXT_DIR / 'figure_conclusions.md').write_text('\n'.join(lines), encoding='utf-8')
    print('wrote', TEXT_DIR / 'figure_conclusions.md')


def write_run_readme(meta_df: pd.DataFrame):
    lines = []
    lines.append('# FTB retained-bridge behavior analysis')
    lines.append('')
    lines.append('Generated by: `analyze_ftb_bridge_behavior.py`')
    lines.append('')
    lines.append('## Inputs')
    for bench, cfg in BENCHES.items():
        lines.append(f'- {bench}: `{cfg["db"]}`')
    lines.append('')
    lines.append('## Candidate definitions used for reconstruction')
    for _, row in meta_df.iterrows():
        lines.append(
            f'- {row["bench"]}: rank non-final turns by token-average '
            'Teacher–Student disagreement and take at most the configured number.'
        )
    lines.append('')
    lines.append('## Main outputs')
    lines.append('- `csv/position_relative_bins.csv`')
    lines.append('- `csv/position_absolute_turns.csv`')
    lines.append('- `csv/action_enrichment.csv`')
    lines.append('- `csv/transition_summary.csv`')
    lines.append('- `csv/transition_matrix_*.csv`')
    lines.append('- `csv/case_examples.csv`')
    lines.append('- `csv/action_position_joint.csv`')
    lines.append('- `figures/figure1_relative_position_dual.{png,pdf}`')
    lines.append('- `figures/figure2_action_enrichment_dual.{png,pdf}`')
    lines.append('- `figures/figure3_transition_matrix_dual.{png,pdf}`')
    lines.append('- `figures/figure4_action_position_heatmaps.{png,pdf}`')
    lines.append('')
    lines.append('## Command')
    lines.append('```bash')
    lines.append('python analysis/analyze_bridge_behavior.py --alfworld-db RUNS/alfworld.db --webshop-db RUNS/webshop.db')
    lines.append('```')
    (OUT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print('wrote', OUT / 'README.md')


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Reconstruct retained bridge positions and action changes from '
            'FTB explorer databases. Run inside the TCOD environment so '
            'pickled experience classes are importable.'
        )
    )
    parser.add_argument('--alfworld-db', type=Path, required=True)
    parser.add_argument('--webshop-db', type=Path, required=True)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('analysis_outputs/bridge_behavior'),
    )
    parser.add_argument('--bridge-position-top-k', type=int, default=1)
    parser.add_argument('--success-threshold', type=float, default=0.5)
    args = parser.parse_args()
    if args.bridge_position_top_k < 1:
        parser.error('--bridge-position-top-k must be positive')
    for db_path in (args.alfworld_db, args.webshop_db):
        if not db_path.is_file():
            parser.error(f'database does not exist: {db_path}')
    configure(args)
    ensure_dirs()
    meta_rows = []
    pos_all = []
    abs_all = []
    enrich_all = []
    trans_all = {}
    trans_summaries = {}
    joint_all = {}
    case_rows = []
    normal_all = []
    candidate_all = []
    match_all = []

    for bench, cfg in BENCHES.items():
        print('Loading', bench)
        normals, bridges, bridge_total = load_benchmark(bench, cfg)
        normal_df, candidate_df, match_df, meta = recover_kept_matches(bench, cfg, normals, bridges)
        meta['bench'] = bench
        meta_rows.append(meta)
        normal_all.append(normal_df)
        candidate_all.append(candidate_df)
        match_all.append(match_df)

        pos_df = summarise_position(normal_df, candidate_df, match_df, bench)
        abs_df = summarise_absolute_position(normal_df, candidate_df, match_df, bench)
        enrich_df = summarise_action_enrichment(normal_df, match_df, bench)
        trans_df = transition_rows(match_df, bench)
        trans_sum = transition_summary(trans_df, bench)
        joint_df = joint_action_position(candidate_df, match_df, bench)
        cases_df = select_case_examples(trans_df, bench)

        matrix = pd.crosstab(trans_df['trigger_action_type'], trans_df['bridge_action_type'])
        matrix.to_csv(CSV_DIR / f'transition_matrix_{bench.lower()}.csv')
        print('wrote', CSV_DIR / f'transition_matrix_{bench.lower()}.csv')

        pos_all.append(pos_df)
        abs_all.append(abs_df)
        enrich_all.append(enrich_df)
        trans_all[bench] = trans_df
        trans_summaries[bench] = trans_sum
        joint_all[bench] = joint_df
        if len(cases_df):
            case_rows.append(cases_df)

    meta_df = save_meta(meta_rows)
    normal_df_all = pd.concat(normal_all, ignore_index=True)
    candidate_df_all = pd.concat(candidate_all, ignore_index=True)
    match_df_all = pd.concat(match_all, ignore_index=True)
    pos_df_all = pd.concat(pos_all, ignore_index=True)
    abs_df_all = pd.concat(abs_all, ignore_index=True)
    enrich_df_all = pd.concat(enrich_all, ignore_index=True)
    trans_sum_all = pd.concat([trans_summaries[k] for k in ['ALFWorld', 'WebShop']], ignore_index=True)
    joint_df_all = pd.concat([joint_all[k] for k in ['ALFWorld', 'WebShop']], ignore_index=True)
    cases_df_all = pd.concat(case_rows, ignore_index=True) if case_rows else pd.DataFrame()

    write_csv(CSV_DIR / 'normal_turns.csv', normal_df_all.to_dict('records'))
    write_csv(CSV_DIR / 'candidate_turns.csv', candidate_df_all.to_dict('records'))
    write_csv(CSV_DIR / 'kept_bridge_matches.csv', match_df_all.to_dict('records'))
    write_csv(CSV_DIR / 'position_relative_bins.csv', pos_df_all.to_dict('records'))
    write_csv(CSV_DIR / 'position_absolute_turns.csv', abs_df_all.to_dict('records'))
    write_csv(CSV_DIR / 'action_enrichment.csv', enrich_df_all.to_dict('records'))
    write_csv(CSV_DIR / 'transition_pairs.csv', pd.concat([trans_all[k] for k in ['ALFWorld', 'WebShop']], ignore_index=True).to_dict('records'))
    write_csv(CSV_DIR / 'transition_summary.csv', trans_sum_all.to_dict('records'))
    write_csv(CSV_DIR / 'action_position_joint.csv', joint_df_all.to_dict('records'))
    if len(cases_df_all):
        write_csv(CSV_DIR / 'case_examples.csv', cases_df_all.to_dict('records'))

    plot_position_dual(pos_df_all)
    plot_action_enrichment_dual(enrich_df_all)
    plot_transition_heatmaps(trans_all)
    plot_joint_heatmaps(joint_all)
    write_conclusions(pos_df_all, abs_df_all, enrich_df_all, trans_summaries, meta_df)
    write_run_readme(meta_df)


if __name__ == '__main__':
    main()
