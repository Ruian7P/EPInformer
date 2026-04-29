#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# ---------- User-tunable defaults ----------
MOTIF_HITS_TSV="${MOTIF_HITS_TSV:-data/promoter_2k_hits_relative.tsv}"
EXPR_ANNOT="${EXPR_ANNOT:-data/GM12878_K562_18377_gene_expr_fromXpresso_with_sequence_strand.csv}"
ATAC_DIR="${ATAC_DIR:-data/atac_peaks}"
DHS_BED="${DHS_BED:-data/K562_ABC_EGLinks/DNase_ENCFF257HEE_Neighborhoods/EnhancerList.bed}"

POS_BINS="${POS_BINS:-8}"
MIN_FEATURE_GENES="${MIN_FEATURE_GENES:-50}"
INCLUDE_GLOBAL="${INCLUDE_GLOBAL:-1}"

OUT_PEAKS="${OUT_PEAKS:-data/atac_peaks/K562_ATAC_top8_union_plus_DHS_EnhancerList.bed}"
OUT_MOTIF="${OUT_MOTIF:-data/motif_bench/promoter_2k_motif_counts_atac_unionDHS_pos8plusglobal_min50.tsv}"
RUN_TRAIN="${RUN_TRAIN:-0}"
SEED="${SEED:-42}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"

echo "[1/4] Build K562 ATAC top8 union peaks (if missing)"
if [[ ! -s "$OUT_PEAKS" ]]; then
  mapfile -t atac_files < <(ls -1 "$ATAC_DIR"/ENCFF*.K562.ATAC*.bed.gz 2>/dev/null | head -n 8)
  if [[ ${#atac_files[@]} -eq 0 ]]; then
    echo "ERROR: no ATAC peak files found under $ATAC_DIR"
    exit 1
  fi
  if ! command -v bedtools >/dev/null 2>&1; then
    echo "ERROR: bedtools not found. Install bedtools to merge peaks."
    exit 1
  fi
  tmp="$(mktemp)"
  for f in "${atac_files[@]}"; do
    zcat "$f" | awk 'BEGIN{OFS="\t"} !/^#/ && NF>=3 {print $1,$2,$3}' >> "$tmp"
  done
  sort -k1,1 -k2,2n "$tmp" | bedtools merge -i - > "${OUT_PEAKS}.atac_only"
  rm -f "$tmp"

  if [[ -s "$DHS_BED" ]]; then
    cat "${OUT_PEAKS}.atac_only" <(awk 'BEGIN{OFS="\t"} !/^#/ && NF>=3 {print $1,$2,$3}' "$DHS_BED") \
      | sort -k1,1 -k2,2n \
      | bedtools merge -i - > "$OUT_PEAKS"
  else
    mv "${OUT_PEAKS}.atac_only" "$OUT_PEAKS"
  fi
else
  echo "Using existing peaks: $OUT_PEAKS"
fi

echo "[2/4] Validate motif hits input"
if [[ ! -s "$MOTIF_HITS_TSV" ]]; then
  echo "ERROR: $MOTIF_HITS_TSV not found."
  echo "Run: bash src/tools/pipeline.sh"
  exit 1
fi

echo "[3/4] Build motif feature matrix with TSS-distance positional bins"
include_flag=()
if [[ "$INCLUDE_GLOBAL" == "1" ]]; then
  include_flag+=(--include-global-motif)
fi

python src/tools/build_openchrom_motif_matrix.py \
  --motif-hits "$MOTIF_HITS_TSV" \
  --hits-sep $'\t' \
  --peaks-bed "$OUT_PEAKS" \
  --out "$OUT_MOTIF" \
  --gene-col gene_id \
  --motif-col motif \
  --start-col start \
  --end-col end \
  --score-col score \
  --agg count \
  --fill-genes-from "$EXPR_ANNOT" \
  --fill-genes-col gene_id \
  --relative-coords \
  --promoter-annot "$EXPR_ANNOT" \
  --promoter-sep "," \
  --promoter-gene-col gene_id \
  --promoter-chrom-col chrom \
  --promoter-start-col start \
  --promoter-end-col end \
  --promoter-strand-col strand \
  --promoter-upstream 1500 \
  --promoter-downstream 500 \
  --position-bins "$POS_BINS" \
  --min-feature-genes "$MIN_FEATURE_GENES" \
  "${include_flag[@]}"

echo "[4/4] Optional training"
if [[ "$RUN_TRAIN" == "1" ]]; then
  mkdir -p logs
  python -u src/train_MoPInformer.py \
    --fold enformer \
    --model_type MoPInformer-P \
    --motif_path "$OUT_MOTIF" \
    --motif_zscore \
    --early_stop_patience "$EARLY_STOP_PATIENCE" \
    --seed "$SEED" \
    | tee "logs/train_MoPInformer_P_atac_tss_pos${POS_BINS}_min${MIN_FEATURE_GENES}_seed${SEED}.log"
fi

echo "Done."
echo "Motif feature file: $OUT_MOTIF"
