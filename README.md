# Hybrid Hyperbolic Graph Reasoning modules (H2GR)
We introduce Hybrid Hyperbolic Graph Reasoning modules (H2GR) that incorporate multi-view semantic correlations into node embeddings via Poincare and Lorentz graph reasoning modules to enhance their semantic representational ability. We also propose an interactive mechanism to gather different features of two modules across multiple layers. To evaluate the performance of H2GR, we conduct experiments on three large-scale recommendation datasets.

<div align=center>
<img src=./fig/framework.png width="100%" ></img>
</div>


## Datasets
We use three widely adopted datasets: Alibaba-iFashion, Yelp2018, and Last-FM for our experiments. Following the preprocessing procedure described in the paper ["Learning Intents behind Interactions with Knowledge Graph for Recommendation"](https://arxiv.org/pdf/2102.07057), we process all raw data accordingly. The processed datasets are stored in the `data/` directory.

## Experiments

Run the experiments with the following commands:

- Alibaba-iFashion dataset
```shell
python main.py --dataset alibaba-ifashion --lr 0.0001 --num_neg_sample 200 --margin 0.6 --node_dropout_rate 0.1 --edge_sampling_rate 0.5
```

- Yelp2018 dataset
```shell
python main.py --dataset yelp2018 --lr 0.0005 --num_neg_sample 400 --margin 0.8 --node_dropout_rate 0.1 --edge_sampling_rate 0.3
```

- Last-FM dataset
```shell
python main.py --dataset last-fm --lr 0.0001 --num_neg_sample 400 --margin 0.7 --node_dropout_rate 0.03 --edge_sampling_rate 0.5
```


