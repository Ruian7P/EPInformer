import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.patches import Patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot representative motif presence (0/1) across selected genes with expression values"
    )
    parser.add_argument("--motif-path", type=str, default="./data/promoter_2k_motif_hits.tsv")
    parser.add_argument(
        "--expr-path",
        type=str,
        default="./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv",
    )
    parser.add_argument("--gene-id-col", type=str, default="gene_id")
    parser.add_argument("--gene-name-col", type=str, default="Gene name")
    parser.add_argument("--expr-col", type=str, default="Actual_K562")
    parser.add_argument("--n-genes", type=int, default=15)
    parser.add_argument("--n-motifs", type=int, default=12)
    parser.add_argument(
        "--min-shown-presence",
        type=int,
        default=2,
        help="Minimum number of displayed motifs required for each selected gene",
    )
    parser.add_argument(
        "--max-shown-presence",
        type=int,
        default=-1,
        help="Maximum number of displayed motifs per selected gene; -1 means n_motifs-1",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="positive",
        choices=["positive", "balanced"],
        help="positive: emphasize motifs/genes with higher expression; balanced: include both positive/negative motifs",
    )
    parser.add_argument(
        "--exclude-families-regex",
        type=str,
        default="Unknown|Mixed",
        help="Regex of motif families to exclude from display for readability",
    )
    parser.add_argument(
        "--presence-thr",
        type=float,
        default=0.0,
        help="Motif value > presence_thr is considered present",
    )
    parser.add_argument("--outdir", type=str, default="./results/motif_pre_model_analysis")
    return parser.parse_args()


def short_gene_label(gene_name: str, gene_id: str) -> str:
    gid = gene_id.replace("ENSG", "")[:6]
    return f"{gene_name}|{gid}"


def pretty_motif_label(motif: str) -> str:
    # Example: GM.5.0.C2H2_ZF.0258 -> C2H2 ZF (0258)
    name = motif
    if name.startswith("GM.5.0."):
        name = name[len("GM.5.0.") :]
    parts = name.split(".")
    if len(parts) >= 2:
        family = parts[0].replace("_", " ")
        idx = parts[1]
        return f"{family} ({idx})"
    return name.replace("_", " ")


def choose_representative_motifs(
    merged: pd.DataFrame,
    expr_col: str,
    n_motifs: int,
    exclude_families_regex: str,
    selection_mode: str,
) -> list[str]:
    motif_cols = [c for c in merged.columns if c not in ["Gene name", expr_col]]
    expr = merged[expr_col].values

    stat_rows = []
    for m in motif_cols:
        x = merged[m].values
        if np.all(x == x[0]):
            continue
        r, p = stats.spearmanr(x, expr)
        if np.isnan(r) or np.isnan(p):
            continue
        stat_rows.append((m, r, p))

    stat_df = pd.DataFrame(stat_rows, columns=["motif", "r", "p"])
    if stat_df.empty:
        return motif_cols[:n_motifs]

    if exclude_families_regex:
        keep_mask = ~stat_df["motif"].str.contains(exclude_families_regex, case=False, regex=True)
        filtered = stat_df[keep_mask]
        if not filtered.empty:
            stat_df = filtered

    # Prefer interpretable family motifs first; fallback to Unknown/Mixed if needed.
    known = stat_df[~stat_df["motif"].str.contains("Unknown|Mixed", case=False, na=False, regex=True)]
    unk = stat_df[stat_df["motif"].str.contains("Unknown|Mixed", case=False, na=False, regex=True)]

    if selection_mode == "positive":
        selected = []
        for pool in [known, unk]:
            for m in pool.sort_values(["r", "p"], ascending=[False, True])["motif"].tolist():
                if m not in selected:
                    selected.append(m)
                if len(selected) >= n_motifs:
                    return selected
        return selected[:n_motifs]

    half = max(1, n_motifs // 2)
    selected = []

    for pool in [known, unk]:
        if len(selected) >= n_motifs:
            break
        pos = pool.sort_values("r", ascending=False).head(half)
        neg = pool.sort_values("r", ascending=True).head(half)
        for m in pd.concat([pos, neg])["motif"].tolist():
            if m not in selected:
                selected.append(m)
            if len(selected) >= n_motifs:
                break

    if len(selected) < n_motifs:
        for m in stat_df.sort_values("p", ascending=True)["motif"].tolist():
            if m not in selected:
                selected.append(m)
            if len(selected) >= n_motifs:
                break

    return selected[:n_motifs]


def choose_diverse_genes(
    df: pd.DataFrame,
    motif_cols: list[str],
    expr_col: str,
    n_genes: int,
    min_shown_presence: int,
) -> pd.DataFrame:
    work = df.copy()
    # Keep genes that show at least one displayed motif to avoid confusing "all-zero" rows.
    work = work[work[motif_cols].sum(axis=1) >= min_shown_presence].copy()
    if work.empty:
        return df.head(n_genes).copy()

    x = work[motif_cols].values.astype(int)
    y = work[expr_col].values.astype(float)

    # Seed with highest and lowest expression genes.
    hi = int(np.argmax(y))
    lo = int(np.argmin(y))
    selected = [hi]
    if lo != hi:
        selected.append(lo)

    expr_std = np.std(y) + 1e-8

    while len(selected) < min(n_genes, len(work)):
        best_i = None
        best_score = -1.0
        for i in range(len(work)):
            if i in selected:
                continue
            # Diversity in motif presence pattern
            hamming_min = min(np.mean(x[i] != x[j]) for j in selected)
            # Also encourage expression spread
            expr_min = min(abs(y[i] - y[j]) / expr_std for j in selected)
            score = hamming_min + 0.25 * expr_min
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)

    out = work.iloc[selected].copy()
    out = out.sort_values(expr_col, ascending=False)
    return out


def choose_positive_pattern_genes(
    df: pd.DataFrame,
    motif_cols: list[str],
    expr_col: str,
    n_genes: int,
    min_shown_presence: int,
    max_shown_presence: int,
) -> pd.DataFrame:
    work = df.copy()
    work["shown_presence_count"] = work[motif_cols].sum(axis=1)
    work = work[work["shown_presence_count"] >= min_shown_presence].copy()
    if max_shown_presence >= 0:
        filtered = work[work["shown_presence_count"] <= max_shown_presence].copy()
        if not filtered.empty:
            work = filtered
    if work.empty:
        return df.head(n_genes).copy()

    # Ranking score used in fill/rebalance steps.
    count_rank_all = work["shown_presence_count"].rank(pct=True)
    expr_rank_all = work[expr_col].rank(pct=True)
    work["score"] = count_rank_all + expr_rank_all

    # Pick two ends to keep the intended positive pattern:
    # high motif-count + high expression, and low motif-count + low expression.
    q_hi = work["shown_presence_count"].quantile(0.65)
    q_lo = work["shown_presence_count"].quantile(0.35)

    hi_pool = work[work["shown_presence_count"] >= q_hi].copy()
    lo_pool = work[work["shown_presence_count"] <= q_lo].copy()

    n_hi = max(1, n_genes // 2)
    n_lo = max(1, n_genes - n_hi)

    hi_pick = hi_pool.sort_values(expr_col, ascending=False).head(min(n_hi, len(hi_pool)))
    lo_pick = lo_pool.sort_values(expr_col, ascending=True).head(min(n_lo, len(lo_pool)))

    out = pd.concat([hi_pick, lo_pick], axis=0).drop_duplicates()

    # Fill remaining slots with genes aligned on count+expression rank.
    if len(out) < min(n_genes, len(work)):
        remain_n = min(n_genes, len(work)) - len(out)
        remain = work.drop(index=out.index, errors="ignore").copy()
        if not remain.empty:
            count_rank = remain["shown_presence_count"].rank(pct=True)
            expr_rank = remain[expr_col].rank(pct=True)
            remain["score"] = count_rank + expr_rank
            fill = remain.sort_values("score", ascending=False).head(remain_n)
            out = pd.concat([out, fill], axis=0)

    # Rebalance for better expression contrast:
    # add 3 genes in [1, 2) and remove 3 genes around -1 (<= -0.8), if possible.
    curr_mid = int(((out[expr_col] >= 1.0) & (out[expr_col] < 2.0)).sum())
    curr_hard_neg = int((out[expr_col] <= -0.8).sum())
    want_mid = min(curr_mid + 3, int(((work[expr_col] >= 1.0) & (work[expr_col] < 2.0)).sum()))
    want_hard_neg = max(0, curr_hard_neg - 3)

    # First, replace hardest negatives with 1-2 candidates.
    while len(out) > 0 and curr_mid < want_mid and curr_hard_neg > want_hard_neg:
        mid_cands = work[
            (work[expr_col] >= 1.0)
            & (work[expr_col] < 2.0)
            & (~work.index.isin(out.index))
        ].sort_values(["shown_presence_count", "score", expr_col], ascending=[False, False, False])
        if mid_cands.empty:
            break
        add_idx = mid_cands.index[0]

        hard_neg_rows = out[out[expr_col] <= -0.8].sort_values(
            ["shown_presence_count", expr_col], ascending=[True, True]
        )
        if hard_neg_rows.empty:
            break
        drop_idx = hard_neg_rows.index[0]

        out = out.drop(index=drop_idx)
        out = pd.concat([out, work.loc[[add_idx]]], axis=0)
        curr_mid = int(((out[expr_col] >= 1.0) & (out[expr_col] < 2.0)).sum())
        curr_hard_neg = int((out[expr_col] <= -0.8).sum())

    # If still lacking 1-2 genes, replace lowest-scoring non-mid genes.
    while len(out) > 0 and curr_mid < want_mid:
        mid_cands = work[
            (work[expr_col] >= 1.0)
            & (work[expr_col] < 2.0)
            & (~work.index.isin(out.index))
        ].sort_values(["shown_presence_count", "score", expr_col], ascending=[False, False, False])
        if mid_cands.empty:
            break
        add_idx = mid_cands.index[0]

        non_mid_rows = out[~((out[expr_col] >= 1.0) & (out[expr_col] < 2.0))].copy()
        if non_mid_rows.empty:
            break
        non_mid_rows = non_mid_rows.join(work[["score"]], how="left")
        drop_idx = non_mid_rows.sort_values(["score", "shown_presence_count", expr_col], ascending=[True, True, True]).index[0]

        out = out.drop(index=drop_idx)
        out = pd.concat([out, work.loc[[add_idx]]], axis=0)
        curr_mid = int(((out[expr_col] >= 1.0) & (out[expr_col] < 2.0)).sum())

    # If hard negatives are still too many, replace extras with best non-hard-negative genes.
    curr_hard_neg = int((out[expr_col] <= -0.8).sum())
    while len(out) > 0 and curr_hard_neg > want_hard_neg:
        rep_cands = work[(work[expr_col] > -0.8) & (~work.index.isin(out.index))].sort_values(
            ["score", "shown_presence_count", expr_col], ascending=[False, False, False]
        )
        if rep_cands.empty:
            break
        add_idx = rep_cands.index[0]
        hard_neg_rows = out[out[expr_col] <= -0.8].sort_values(
            ["shown_presence_count", expr_col], ascending=[True, True]
        )
        if hard_neg_rows.empty:
            break
        drop_idx = hard_neg_rows.index[0]

        out = out.drop(index=drop_idx)
        out = pd.concat([out, work.loc[[add_idx]]], axis=0)
        curr_hard_neg = int((out[expr_col] <= -0.8).sum())

    # Remove duplicate motif patterns (e.g., visually identical gene rows), then refill.
    pat_cols = motif_cols
    out = out.drop_duplicates(subset=pat_cols, keep="first")

    if len(out) < min(n_genes, len(work)):
        remain_n = min(n_genes, len(work)) - len(out)
        remain = work.drop(index=out.index, errors="ignore").copy()
        if not remain.empty:
            existing_patterns = set(tuple(v) for v in out[pat_cols].astype(int).values.tolist())

            # First add candidates with new motif patterns.
            remain_sorted = remain.sort_values(["score", "shown_presence_count", expr_col], ascending=[False, False, False])
            add_indices = []
            for idx, row in remain_sorted.iterrows():
                pat = tuple(row[pat_cols].astype(int).tolist())
                if pat in existing_patterns:
                    continue
                add_indices.append(idx)
                existing_patterns.add(pat)
                if len(add_indices) >= remain_n:
                    break

            # If unique patterns are insufficient, fill remaining slots by score.
            if len(add_indices) < remain_n:
                for idx in remain_sorted.index.tolist():
                    if idx in add_indices:
                        continue
                    add_indices.append(idx)
                    if len(add_indices) >= remain_n:
                        break

            if add_indices:
                out = pd.concat([out, work.loc[add_indices]], axis=0)

    # User-facing contrast tweak: reduce one near -1 gene and add one 0-1 gene, if feasible.
    curr_hard_neg = int((out[expr_col] <= -0.8).sum())
    curr_0_1 = int(((out[expr_col] >= 0.0) & (out[expr_col] < 1.0)).sum())
    if curr_hard_neg > 0:
        rep_cands = work[
            (work[expr_col] >= 0.0)
            & (work[expr_col] < 1.0)
            & (~work.index.isin(out.index))
        ].copy()
        if not rep_cands.empty:
            # Prefer a replacement with a new motif pattern if possible.
            existing_patterns = set(tuple(v) for v in out[pat_cols].astype(int).values.tolist())
            rep_cands = rep_cands.sort_values(["score", "shown_presence_count", expr_col], ascending=[False, False, False])
            add_idx = None
            for idx, row in rep_cands.iterrows():
                pat = tuple(row[pat_cols].astype(int).tolist())
                if pat not in existing_patterns:
                    add_idx = idx
                    break
            if add_idx is None:
                add_idx = rep_cands.index[0]

            hard_neg_rows = out[out[expr_col] <= -0.8].copy()
            if "score" not in hard_neg_rows.columns:
                hard_neg_rows = hard_neg_rows.join(work[["score"]], how="left")
            drop_idx = hard_neg_rows.sort_values(["score", "shown_presence_count", expr_col], ascending=[True, True, True]).index[0]

            out = out.drop(index=drop_idx)
            out = pd.concat([out, work.loc[[add_idx]]], axis=0)

    out = out.sort_values(expr_col, ascending=False)
    return out.drop(columns=[c for c in ["score"] if c in out.columns])


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    motif = pd.read_csv(args.motif_path, sep="\t", comment="#", index_col=0, engine="python")
    motif = motif.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    expr = pd.read_csv(args.expr_path)
    needed = [args.gene_id_col, args.gene_name_col, args.expr_col]
    for c in needed:
        if c not in expr.columns:
            raise KeyError(f"{c} not found in expression table")
    expr = expr[needed].dropna().drop_duplicates(subset=[args.gene_id_col]).set_index(args.gene_id_col)

    merged = motif.join(expr, how="inner")
    if merged.empty:
        raise ValueError("No overlapping genes between motif and expression tables")

    selected_motifs = choose_representative_motifs(
        merged,
        args.expr_col,
        args.n_motifs,
        args.exclude_families_regex,
        args.selection_mode,
    )

    # Build binary presence matrix
    binary = (merged[selected_motifs] > args.presence_thr).astype(int)
    binary[args.gene_name_col] = merged[args.gene_name_col]
    binary[args.expr_col] = merged[args.expr_col]

    if args.selection_mode == "positive":
        max_shown_presence = args.max_shown_presence
        if max_shown_presence < 0:
            max_shown_presence = max(args.min_shown_presence, args.n_motifs - 1)
        selected_genes = choose_positive_pattern_genes(
            binary,
            selected_motifs,
            args.expr_col,
            args.n_genes,
            args.min_shown_presence,
            max_shown_presence,
        )
    else:
        selected_genes = choose_diverse_genes(
            binary,
            selected_motifs,
            args.expr_col,
            args.n_genes,
            args.min_shown_presence,
        )

    # Always present genes from high expression to low expression.
    selected_genes = selected_genes.sort_values(args.expr_col, ascending=False)

    # Reorder motifs by usage among selected genes (high -> low) for readability.
    motif_usage = selected_genes[selected_motifs].sum(axis=0).sort_values(ascending=False)
    selected_motifs = motif_usage.index.tolist()

    # Save selected data tables
    selected_genes_out = selected_genes[[args.gene_name_col, args.expr_col] + selected_motifs].copy()
    selected_genes_out.index.name = args.gene_id_col
    selected_genes_out.to_csv(outdir / "selected_genes_binary_motif_presence.csv")

    selected_genes_out["shown_presence_count"] = selected_genes[selected_motifs].sum(axis=1).values
    rho, pval = stats.spearmanr(selected_genes_out["shown_presence_count"].values, selected_genes_out[args.expr_col].values)
    print(f"Selected genes Spearman(count, expression) = {rho:.3f}, p = {pval:.2e}")

    pd.DataFrame({"motif": selected_motifs}).to_csv(
        outdir / "selected_representative_motifs.csv", index=False
    )

    # Plot: left heatmap for motif presence, right bar for expression
    heat = selected_genes[selected_motifs].values.astype(int)
    expr_vals = selected_genes[args.expr_col].values.astype(float)
    gene_ids = selected_genes.index.to_numpy()
    gene_names = selected_genes[args.gene_name_col].fillna("NA").astype(str).to_numpy()
    ylabels = [short_gene_label(gn, gid) for gn, gid in zip(gene_names, gene_ids)]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.6], wspace=0.08)

    ax_h = fig.add_subplot(gs[0, 0])
    im = ax_h.imshow(heat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    motif_labels = [pretty_motif_label(m) for m in selected_motifs]
    ax_h.set_xticks(np.arange(len(selected_motifs)))
    ax_h.set_xticklabels(motif_labels, rotation=70, ha="right", fontsize=8)
    ax_h.set_yticks(np.arange(len(ylabels)))
    ax_h.set_yticklabels(ylabels, fontsize=9)
    ax_h.set_title("Motif Presence (1=present, 0=absent)")
    ax_h.set_xlabel("Motifs")
    ax_h.set_ylabel("Genes")

    # legend_handles = [
    #     Patch(facecolor=plt.cm.Blues(0.05), edgecolor="none", label="Absent (0)"),
    #     Patch(facecolor=plt.cm.Blues(0.8), edgecolor="none", label="Present (1)"),
    # ]
    # ax_h.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=False)

    ax_e = fig.add_subplot(gs[0, 1], sharey=ax_h)
    y = np.arange(len(ylabels))
    ax_e.barh(y, expr_vals, color="#E07A5F")
    ax_e.set_xlabel(f"Gene Expression Label")
    ax_e.set_title("Expression")
    ax_e.tick_params(axis="y", left=False, labelleft=False)

    for ax in [ax_h, ax_e]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_png = outdir / "representative_motif_presence_15genes.png"
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print("Saved plot:", out_png)
    print("Saved gene x motif matrix:", outdir / "selected_genes_binary_motif_presence.csv")
    print("Saved selected motifs:", outdir / "selected_representative_motifs.csv")


if __name__ == "__main__":
    main()
