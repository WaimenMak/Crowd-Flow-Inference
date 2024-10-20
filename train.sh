#python Online_Update.py --model_type="Online_LSTM" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad
#
#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad
#
#python Online_Update.py --model_type="Online_MA" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad

#python Online_Update.py --model_type="Online_GAT" --lags=5 --chunk_size=15 --pred_horizons=7 --train_steps=200 --dataset=crossroad

#python Online_Update.py --model_type="Online_GAT" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad
#
#python Online_Update.py --model_type="Online_GAT" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_GAT" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=train_station
##
#python Online_Update.py --model_type="Online_LSTM" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad
#
#python Online_Update.py --model_type="Online_LSTM" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_LSTM" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=train_station
##
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=train_station
#
#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad
#
#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=train_station


# vary the chunk size
clear

python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=15 --pred_horizons=7 --train_steps=200 --dataset=crossroad

python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=crossroad

python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=45 --pred_horizons=7 --train_steps=200 --dataset=crossroad

python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=60 --pred_horizons=7 --train_steps=200 --dataset=crossroad

python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=75 --pred_horizons=7 --train_steps=200 --dataset=crossroad

python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=90 --pred_horizons=7 --train_steps=200 --dataset=crossroad

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=120 --pred_horizons=7 --train_steps=200 --dataset=crossroad

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=150 --pred_horizons=7 --train_steps=200 --dataset=crossroad

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=180 --pred_horizons=7 --train_steps=200 --dataset=crossroad

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=0 --dataset=crossroad

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=15 --pred_horizons=7 --train_steps=200 --dataset=train_station
#
#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=train_station

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=45 --pred_horizons=7 --train_steps=200 --dataset=train_station

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=60 --pred_horizons=7 --train_steps=200 --dataset=train_station

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=75 --pred_horizons=7 --train_steps=200 --dataset=train_station

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=90 --pred_horizons=7 --train_steps=200 --dataset=train_station

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=120 --pred_horizons=7 --train_steps=200 --dataset=train_station

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=150 --pred_horizons=7 --train_steps=200 --dataset=train_station

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=180 --pred_horizons=7 --train_steps=200 --dataset=train_station

#python Online_Update.py --model_type="Online_Diffusion" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=0 --dataset=train_station

#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=150 --pred_horizons=7 --train_steps=200 --dataset=crossroad
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=180 --pred_horizons=7 --train_steps=200 --dataset=crossroad
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=0 --dataset=crossroad --no-save
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=15 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=45 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=60 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=75 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=90 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=120 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=150 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=180 --pred_horizons=7 --train_steps=200 --dataset=maze
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=0 --dataset=maze --no-save
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=15 --pred_horizons=7 --train_steps=200 --dataset=train_station
##
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=200 --dataset=train_station
##
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=45 --pred_horizons=7 --train_steps=200 --dataset=train_station
##
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=60 --pred_horizons=7 --train_steps=200 --dataset=train_station
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=75 --pred_horizons=7 --train_steps=200 --dataset=train_station
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=90 --pred_horizons=7 --train_steps=200 --dataset=train_station
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=120 --pred_horizons=7 --train_steps=200 --dataset=train_station
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=150 --pred_horizons=7 --train_steps=200 --dataset=train_station
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=180 --pred_horizons=7 --train_steps=200 --dataset=train_station
#
#python Online_Update.py --model_type="Online_GCN" --lags=5 --chunk_size=30 --pred_horizons=7 --train_steps=0 --dataset=train_station --no-save


