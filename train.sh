python Online_Update.py --model_type="Online_LSTM" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad

python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad

python Online_Update.py --model_type="Online_MA" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad

python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad

python Online_Update.py --model_type="Online_GAT" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad