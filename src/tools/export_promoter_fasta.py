import pandas as pd

in_csv = "./data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv"
out_fa = "./data/promoter_2k.fa"

df = pd.read_csv(in_csv, index_col="gene_id")
seqs = df["promoter_2k"].dropna()

with open(out_fa, "w") as f:
    for gid, seq in seqs.items():
        s = str(seq).upper().replace(" ", "").replace("\n", "")
        if s:
            f.write(f">{gid}\n{s}\n")

print("n_promoters =", len(seqs))
print("saved =", out_fa)