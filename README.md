# EMTSF

This is a PyTorch implementation of the paper: Evolutionary Neural Architecture Search for Multivariate Time Series Forecasting.

## Data Preparation

### Multivariate time series datasets

Download Exchange-rate, Electricity datasets from [https://github.com/LiuZH-19/ESG](https://github.com/LiuZH-19/ESG). Move them into the data folder.

### Traffic datasets

Download the METR-LA and PEMS-BAY dataset from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git) . Move them into the data folder.

```

# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```
# Requirements
python 3.8

pytorch 2.0

DGL 1.0.1

## Architecture Search

### Single-step forecasting

```
nohup python -u new_str2genotype.py --data "exchange-rate" --horizon 3 --mulity False > search_record.out 2>&1 &
```

### Multi-step forecasting

```
nohup python -u new_str2genotype.py --data "PEMS-BAY"  --mulity True > search_record.out 2>&1 &
```
