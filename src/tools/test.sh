python -u src/train_MoPInformer.py \
  --fold all \
  --model_type MoPInformer-P \
  --motif_path /home/ruian7p/Projects/EPInformer/data/motif_bench/promoter_2k_motif_counts_atac_unionDHS_pos8plusglobal_min50.tsv \
  --motif_zscore \
  --early_stop_patience 10 \
  --seed 42 \
  | tee logs/train_MoPInformer_P_atac_pos8plusglobal_min50_10.log

python -u src/train_EPInformer_abc.py --fold all --model_type EPInformer-promoter-v2 --early_stop_patience 10 --seed 42 | tee logs/train_epinformerv2_promoter_10.log 


python -u src/train_EPInformer_abc.py --fold all --model_type EPInformer-v2 --early_stop_patience 10 --seed 42 | tee logs/train_epinformerv2_pe_10.log 


python -u src/train_MoPInformer.py --fold all --model_type MoPInformer --motif_path /home/ruian7p/Projects/EPInformer/data/motif_bench/promoter_2k_motif_counts_atac_unionDHS_pos8plusglobal_min50.tsv --motif_zscore --early_stop_patience 10 | tee logs/train_MoPInformer_atac_pos8plusglobal_min50_10.log
