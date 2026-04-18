import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build gene x motif matrix after open-chromatin filtering. "
            "Only motif hits overlapping peaks are kept."
        )
    )
    parser.add_argument("--motif-hits", type=str, required=True, help="Input motif hits table (TSV/CSV)")
    parser.add_argument("--hits-sep", type=str, default="\t", help="Delimiter for motif hit table")
    parser.add_argument("--peaks-bed", type=str, required=True, help="Open chromatin peaks BED file")
    parser.add_argument("--out", type=str, required=True, help="Output gene x motif matrix TSV")

    parser.add_argument("--chrom-col", type=str, default="chrom")
    parser.add_argument("--start-col", type=str, default="start")
    parser.add_argument("--end-col", type=str, default="end")
    parser.add_argument("--gene-col", type=str, default="gene_id")
    parser.add_argument("--motif-col", type=str, default="motif")
    parser.add_argument("--score-col", type=str, default="score")

    parser.add_argument(
        "--agg",
        type=str,
        default="count",
        choices=["count", "maxscore"],
        help="How to aggregate filtered hits into gene x motif values",
    )
    parser.add_argument(
        "--fill-genes-from",
        type=str,
        default=None,
        help="Optional table containing all genes to include as zero rows when missing",
    )
    parser.add_argument(
        "--fill-genes-col",
        type=str,
        default="gene_id",
        help="Gene column in --fill-genes-from",
    )

    parser.add_argument(
        "--relative-coords",
        action="store_true",
        help=(
            "Interpret hit start/end as positions relative to promoter sequence and convert "
            "to genomic coordinates via promoter annotation"
        ),
    )
    parser.add_argument(
        "--promoter-annot",
        type=str,
        default=None,
        help=(
            "Required if --relative-coords. CSV/TSV with gene_id/chrom/start/end/strand to infer promoter window"
        ),
    )
    parser.add_argument("--promoter-sep", type=str, default=",")
    parser.add_argument("--promoter-gene-col", type=str, default="gene_id")
    parser.add_argument("--promoter-chrom-col", type=str, default="chrom")
    parser.add_argument("--promoter-start-col", type=str, default="start")
    parser.add_argument("--promoter-end-col", type=str, default="end")
    parser.add_argument("--promoter-strand-col", type=str, default="strand")
    parser.add_argument(
        "--promoter-upstream",
        type=int,
        default=1500,
        help="Promoter upstream size from TSS for sequence used in motif scan",
    )
    parser.add_argument(
        "--promoter-downstream",
        type=int,
        default=500,
        help="Promoter downstream size from TSS for sequence used in motif scan",
    )
    return parser.parse_args()


def load_peaks_and_merge(peaks_bed: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    peaks = pd.read_csv(
        peaks_bed,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )
    peaks = peaks.dropna()
    peaks["start"] = peaks["start"].astype(int)
    peaks["end"] = peaks["end"].astype(int)
    peaks = peaks[peaks["end"] > peaks["start"]]

    merged = {}
    for chrom, g in peaks.groupby("chrom", sort=False):
        arr = g[["start", "end"]].sort_values(["start", "end"]).to_numpy(dtype=np.int64)
        if arr.size == 0:
            continue
        out = [arr[0].tolist()]
        for s, e in arr[1:]:
            if s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], int(e))
            else:
                out.append([int(s), int(e)])
        out = np.asarray(out, dtype=np.int64)
        merged[chrom] = (out[:, 0], out[:, 1])
    return merged


def overlap_mask_for_chrom(starts: np.ndarray, ends: np.ndarray, peak_starts: np.ndarray, peak_ends: np.ndarray) -> np.ndarray:
    # For merged (non-overlapping) peaks: hit overlaps if peak_start < hit_end and peak_end > hit_start.
    idx = np.searchsorted(peak_starts, ends, side="right") - 1
    valid = idx >= 0
    out = np.zeros(len(starts), dtype=bool)
    if valid.any():
        iv = idx[valid]
        out_valid = peak_ends[iv] > starts[valid]
        out[np.where(valid)[0]] = out_valid
    return out


def ensure_columns(df: pd.DataFrame, cols: list[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {what}: {missing}")


def convert_relative_to_genomic(hits: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.promoter_annot is None:
        raise ValueError("--promoter-annot is required when --relative-coords is set")

    promoter = pd.read_csv(args.promoter_annot, sep=args.promoter_sep)
    ensure_columns(
        promoter,
        [
            args.promoter_gene_col,
            args.promoter_chrom_col,
            args.promoter_start_col,
            args.promoter_end_col,
            args.promoter_strand_col,
        ],
        "promoter annotation",
    )

    pa = promoter[
        [
            args.promoter_gene_col,
            args.promoter_chrom_col,
            args.promoter_start_col,
            args.promoter_end_col,
            args.promoter_strand_col,
        ]
    ].copy()
    pa = pa.drop_duplicates(subset=[args.promoter_gene_col])
    pa[args.promoter_start_col] = pa[args.promoter_start_col].astype(int)
    pa[args.promoter_end_col] = pa[args.promoter_end_col].astype(int)

    # TSS inference from gene interval + strand.
    strand = pa[args.promoter_strand_col].astype(str)
    tss = np.where(strand.values == "-", pa[args.promoter_end_col].values, pa[args.promoter_start_col].values)
    prom_start = tss - int(args.promoter_upstream)
    prom_end = tss + int(args.promoter_downstream)
    pa = pa.assign(_prom_start=prom_start, _prom_end=prom_end)

    # Use fixed internal names to avoid merge collisions with hit columns
    # (for example, both tables can have a "strand" column).
    pa = pa.rename(
        columns={
            args.promoter_gene_col: "_pa_gene",
            args.promoter_chrom_col: "_pa_chrom",
            args.promoter_start_col: "_pa_start",
            args.promoter_end_col: "_pa_end",
            args.promoter_strand_col: "_pa_strand",
        }
    )
    pa = pa[["_pa_gene", "_pa_chrom", "_pa_start", "_pa_end", "_pa_strand", "_prom_start", "_prom_end"]]

    merged = hits.merge(
        pa,
        left_on=args.gene_col,
        right_on="_pa_gene",
        how="left",
    )
    merged = merged.dropna(subset=["_prom_start", "_prom_end", "_pa_chrom", "_pa_strand"])

    rel_start = merged[args.start_col].astype(int).values
    rel_end = merged[args.end_col].astype(int).values
    pstart = merged["_prom_start"].astype(int).values
    pend = merged["_prom_end"].astype(int).values
    s = merged["_pa_strand"].astype(str).values

    abs_start = np.where(s == "-", pend - rel_end, pstart + rel_start)
    abs_end = np.where(s == "-", pend - rel_start, pstart + rel_end)

    out = merged.copy()
    out[args.chrom_col] = merged["_pa_chrom"].values
    out[args.start_col] = abs_start.astype(np.int64)
    out[args.end_col] = abs_end.astype(np.int64)
    out = out[out[args.end_col] > out[args.start_col]]
    return out


def main() -> None:
    args = parse_args()

    hits = pd.read_csv(args.motif_hits, sep=args.hits_sep)
    ensure_columns(hits, [args.gene_col, args.motif_col, args.start_col, args.end_col], "motif hits")

    if args.relative_coords:
        hits = convert_relative_to_genomic(hits, args)
    else:
        ensure_columns(hits, [args.chrom_col], "motif hits")

    hits[args.start_col] = hits[args.start_col].astype(int)
    hits[args.end_col] = hits[args.end_col].astype(int)
    hits = hits[hits[args.end_col] > hits[args.start_col]]
    hits[args.chrom_col] = hits[args.chrom_col].astype(str)

    peaks = load_peaks_and_merge(args.peaks_bed)

    # Auto-normalize chromosome naming between hits and peaks (chr1 vs 1).
    if len(peaks) > 0 and len(hits) > 0:
        peak_has_chr = str(next(iter(peaks.keys()))).startswith("chr")
        hit_has_chr = hits[args.chrom_col].iloc[0].startswith("chr")

        if peak_has_chr and not hit_has_chr:
            hits[args.chrom_col] = "chr" + hits[args.chrom_col]
        elif not peak_has_chr and hit_has_chr:
            hits[args.chrom_col] = hits[args.chrom_col].str.replace(r"^chr", "", regex=True)

    keep = np.zeros(len(hits), dtype=bool)
    for chrom, idx in hits.groupby(args.chrom_col, sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        if chrom not in peaks:
            continue
        ps, pe = peaks[chrom]
        hs = hits.iloc[idx][args.start_col].to_numpy(dtype=np.int64)
        he = hits.iloc[idx][args.end_col].to_numpy(dtype=np.int64)
        mask = overlap_mask_for_chrom(hs, he, ps, pe)
        keep[idx] = mask

    filtered = hits.loc[keep].copy()

    if filtered.empty:
        print("Warning: no motif hits overlap peaks; output will be all-zero if --fill-genes-from is provided.")

    if args.agg == "count":
        vals = filtered.groupby([args.gene_col, args.motif_col]).size().rename("value").reset_index()
    else:
        ensure_columns(filtered, [args.score_col], "motif hits for maxscore aggregation")
        vals = (
            filtered.groupby([args.gene_col, args.motif_col])[args.score_col]
            .max()
            .rename("value")
            .reset_index()
        )

    if vals.empty:
        mat = pd.DataFrame(index=pd.Index([], name=args.gene_col))
    else:
        mat = vals.pivot(index=args.gene_col, columns=args.motif_col, values="value").fillna(0.0)
        mat = mat.sort_index(axis=0).sort_index(axis=1)

    if args.fill_genes_from is not None:
        all_genes_df = pd.read_csv(args.fill_genes_from)
        ensure_columns(all_genes_df, [args.fill_genes_col], "--fill-genes-from")
        all_genes = all_genes_df[args.fill_genes_col].dropna().astype(str).drop_duplicates().tolist()
        if len(all_genes) > 0:
            mat = mat.reindex(all_genes, fill_value=0.0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat.to_csv(out_path, sep="\t")

    total_hits = len(hits)
    kept_hits = len(filtered)
    frac = 0.0 if total_hits == 0 else kept_hits / total_hits
    print(f"Input hits: {total_hits}")
    print(f"Hits overlapping peaks: {kept_hits} ({frac:.2%})")
    print(f"Output matrix: {out_path}")
    print(f"Genes x motifs: {mat.shape[0]} x {mat.shape[1]}")


if __name__ == "__main__":
    main()
