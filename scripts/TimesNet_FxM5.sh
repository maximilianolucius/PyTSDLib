export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --valid_date 2024-01-01 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/forex/\
  --data_path forex_M5.csv \
  --model_id TimesNet_FxM5_96_96 \
  --model $model_name \
  --data FxM5 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --itr 1