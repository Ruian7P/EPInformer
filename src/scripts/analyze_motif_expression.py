import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
	pvals = np.asarray(pvals, dtype=float)
	n = pvals.size
	order = np.argsort(pvals)
	ranked = pvals[order]
	q = ranked * n / (np.arange(n) + 1)
	q = np.minimum.accumulate(q[::-1])[::-1]
	q = np.clip(q, 0.0, 1.0)
	out = np.empty_like(q)
	out[order] = q
	return out


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Pre-model analysis: motif feature differences vs gene expression"
	)
	parser.add_argument(
		"--motif-path",
		type=str,
		default="./data/promoter_2k_motif_hits.tsv",
		help="Path to motif feature table (rows=genes, cols=motifs)",
	)
	parser.add_argument(
		"--expr-path",
		type=str,
		default="./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv",
		help="Path to gene expression table",
	)
	parser.add_argument(
		"--expr-col",
		type=str,
		default="Actual_K562",
		help="Expression column to analyze, e.g. Actual_K562 or Actual_GM12878",
	)
	parser.add_argument(
		"--gene-id-col",
		type=str,
		default="gene_id",
		help="Gene ID column name in expression table",
	)
	parser.add_argument(
		"--top-frac",
		type=float,
		default=0.25,
		help="Top/bottom expression fraction for differential motif analysis",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=30,
		help="How many top motifs to export for each analysis",
	)
	parser.add_argument(
		"--outdir",
		type=str,
		default="./results/motif_pre_model_analysis",
		help="Output directory",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	motif_df = pd.read_csv(
		args.motif_path,
		sep="\t",
		comment="#",
		index_col=0,
		engine="python",
	)
	motif_df = motif_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

	expr_df = pd.read_csv(args.expr_path)
	if args.gene_id_col not in expr_df.columns:
		raise KeyError(f"{args.gene_id_col} not found in expression file")
	if args.expr_col not in expr_df.columns:
		raise KeyError(f"{args.expr_col} not found in expression file")

	expr_sub = expr_df[[args.gene_id_col, args.expr_col]].copy()
	expr_sub = expr_sub.dropna()
	expr_sub = expr_sub.drop_duplicates(subset=[args.gene_id_col])
	expr_sub = expr_sub.set_index(args.gene_id_col)

	merged = motif_df.join(expr_sub, how="inner")
	if merged.empty:
		raise ValueError("No overlapping genes between motif table and expression table")

	merged.to_csv(outdir / "merged_motif_expression_matrix.csv")

	expr = merged[args.expr_col].astype(float).values
	motif_mat = merged.drop(columns=[args.expr_col]).values
	motif_names = merged.drop(columns=[args.expr_col]).columns.to_numpy()

	# 1) Per-motif Spearman correlation with expression
	r_list = np.zeros(motif_mat.shape[1], dtype=float)
	p_list = np.ones(motif_mat.shape[1], dtype=float)
	for i in range(motif_mat.shape[1]):
		xi = motif_mat[:, i]
		if np.all(xi == xi[0]):
			r_list[i] = 0.0
			p_list[i] = 1.0
			continue
		r, p = stats.spearmanr(xi, expr)
		r_list[i] = 0.0 if np.isnan(r) else r
		p_list[i] = 1.0 if np.isnan(p) else p

	q_list = benjamini_hochberg(p_list)
	corr_df = pd.DataFrame(
		{
			"motif": motif_names,
			"spearman_r": r_list,
			"p_value": p_list,
			"fdr_bh": q_list,
		}
	).sort_values("spearman_r", ascending=False)
	corr_df.to_csv(outdir / "motif_expression_spearman_all.csv", index=False)

	top_pos = corr_df.head(args.top_k)
	top_neg = corr_df.tail(args.top_k).sort_values("spearman_r", ascending=True)
	top_pos.to_csv(outdir / "top_positive_motifs_by_spearman.csv", index=False)
	top_neg.to_csv(outdir / "top_negative_motifs_by_spearman.csv", index=False)

	# 2) Differential motif abundance: high-expression vs low-expression genes
	n = len(merged)
	group_n = max(1, int(n * args.top_frac))
	expr_rank = merged[args.expr_col].rank(method="first")
	low_mask = expr_rank <= group_n
	high_mask = expr_rank > (n - group_n)

	high_vals = motif_mat[high_mask.values]
	low_vals = motif_mat[low_mask.values]

	mean_high = high_vals.mean(axis=0)
	mean_low = low_vals.mean(axis=0)
	diff = mean_high - mean_low

	p_diff = np.ones(motif_mat.shape[1], dtype=float)
	for i in range(motif_mat.shape[1]):
		try:
			_, p = stats.mannwhitneyu(high_vals[:, i], low_vals[:, i], alternative="two-sided")
			p_diff[i] = p
		except ValueError:
			p_diff[i] = 1.0

	q_diff = benjamini_hochberg(p_diff)
	diff_df = pd.DataFrame(
		{
			"motif": motif_names,
			"mean_high": mean_high,
			"mean_low": mean_low,
			"mean_diff_high_minus_low": diff,
			"p_value": p_diff,
			"fdr_bh": q_diff,
		}
	).sort_values("mean_diff_high_minus_low", ascending=False)
	diff_df.to_csv(outdir / "motif_high_vs_low_expression_diff.csv", index=False)

	# 3) Plots
	plt.figure(figsize=(7, 5))
	plt.hist(expr, bins=60)
	plt.xlabel(args.expr_col)
	plt.ylabel("Gene count")
	plt.title("Expression distribution")
	plt.tight_layout()
	plt.savefig(outdir / "expression_distribution.png", dpi=220)
	plt.close()

	plot_df = corr_df.copy()
	plot_df["neg_log10_p"] = -np.log10(np.clip(plot_df["p_value"].values, 1e-300, 1.0))
	plt.figure(figsize=(7, 5))
	plt.scatter(plot_df["spearman_r"], plot_df["neg_log10_p"], s=8, alpha=0.35)
	plt.xlabel("Spearman r (motif vs expression)")
	plt.ylabel("-log10(p-value)")
	plt.title("Motif-expression association")
	plt.tight_layout()
	plt.savefig(outdir / "motif_expression_volcano_like.png", dpi=220)
	plt.close()

	print("Analysis complete")
	print(f"Matched genes: {len(merged)}")
	print(f"Top positive motifs saved to: {outdir / 'top_positive_motifs_by_spearman.csv'}")
	print(f"Top negative motifs saved to: {outdir / 'top_negative_motifs_by_spearman.csv'}")
	print(f"Differential motifs saved to: {outdir / 'motif_high_vs_low_expression_diff.csv'}")


if __name__ == "__main__":
	main()
