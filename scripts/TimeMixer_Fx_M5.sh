export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=16

python -u run.py \
  --valid_date 2024-01-01 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/forex/\
  --data_path forex_M5.csv \
  --model_id FxM5_$seq_len'_'96 \
  --model $model_name \
  --data FxM5 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

