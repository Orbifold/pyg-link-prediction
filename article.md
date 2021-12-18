
## A note on Pyg Datasets

It's not strictly necessary to create custom Pyg datasets but doing so comes with some pleasant advantages:

- caching of the dataset (as [a Torch tensor file](https://pytorch.org/docs/stable/generated/torch.save.html)) is automatic and after the raw data has been downloaded and transformed, Pyg will subsequently load the cached file
- the [DataLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.DataLoader) and [NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader) in particular require a Dataset to function
- [Pyg transformation](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html) operate on datasets.

Creating a custom Pyg dataset using the raw Pokec data is not complicated and the details can be seen [here](https://github.com/Orbifold/pyg-link-prediction/blob/main/pokec/Pokec.py). The main ingredients are

- downloading the raw data
- transforming the nodes and edges to Torch compatibles structures (edge index, Torch tensors and so on)
- instantiating a [Data](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data) object if you want to work with a homogeneous graph or [HeteroData](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.HeteroData) for heterogeneous graphs.

Transforming the data is really a standard exercise with Pandas dataframes, Numpy/Torch tensorizations and the typical data science data wrangling. The process defines to some extent the type of machine learning you can perform in the sense that one obviously can't learn from what is not present. If you don't encode a node payload into Torch tensors you can't learn from it (e.g. node embeddings and classifications).

Once the data is wrapped it's now one line to fetch the data

```python
pokec = Pokec()
```
and as simple as this

```python
transform = T.Compose([
    T.ToUndirected(merge = True),  
    T.RandomLinkSplit(num_val = 0.0005, num_test = 0.0001, is_undirected = True, add_negative_train_samples = False)
])
train_data, val_data, test_data = transform(data)
```

to create the necessary train-validation-test triple.

## Dealing with big graphs

The Pokec network might not be characterized as "big data" (anything not fitting on a single disk), it's nevertheless too big to use as a single entity towards graph machine learning. Pyg will not complain but will not return either, it simply hangs. To alleviate this Pyg has a sampler based on the [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) paper and it, in essence, allows you to learn from overlapping subgraphs. Since learning involves the typical message passing mechanism of graph convolutions, this is does not represent basic graph sampling but the details are beyond the scope of this article.

In practice, the only thing you need to do is to define a loader and loop over the generated batches. In the case of the training loop this is all you need:

```python
train_loader = NeighborLoader(data, input_nodes = pokec.data.train_mask, batch_size = 128)
```
The [NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader) takes the whole dataset but samples from it via the masks. These masks are created automatically when splitting the data as shown above. The batch size defines the subgraph radius and the larger the value the faster you will train the model. It depends on your hardware and, well, patience.

Different loaders are defined for the train, test and validation sets. They can have different parametrizations but in practice one keeps them equal.

The batch objects can be approached in the same way as a Data object. For example, you can inspect the `edge_index`, node payload `batch.x` and node count `batch.num_nodes` just like a non-sampled set

```python
for batch in train_loader:
    print(batch.x)
    print(batch.edge_index)
    print(batch.num_nodes)
```

## Learning what is there and what is not

With the big graph handling out of the way one can focus on the actual learning and the neural net definition. The latter is quite easy thanks to [Pyg's rich set of layers](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html). Like any other machine learning this is where science and experience come together, but for ourpurposes the neural net is just a couple of graph convolutions:

```python
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
```
The effect of processing a graph (i.e. the training loop) through this net is that connected nodes will be nearer each other in the latent space than nodes that are not. Near means here that a dot-product (aka cosine similarity) of the embeddings is an indication of connectedness.

This would be the end of the story if we would observe a classification task but in the case of edge completion there is the issue that learning from the graph incidence structure effectively means learning only from existing edges. It corresponds to an imbalanced dataset in standard machine learning. The way to balance the data is by sampling couple of nodes which are not connected, the so-called negative edges. For each batch (subgraph) we take the `batch.edge_index` and augment it with negative edges sampled via the `negative_sampling` method provided by Pyg:

```python
neg_edge_index = negative_sampling(edge_index = batch.edge_index, num_nodes = batch.num_nodes, num_neg_samples = None)
```
By setting `num_neg_samples` to `None` we tell the sampling process to pick up as many negative edges as there are positive ones in the batch, resulting in a perfectly balanced dataset.

With this in place, the training process will learn a function which maps couple of nodes to one or zero corresponding to, respectively, whether this couple has an edge or not. Since the range of this function is binary it's hence natural to use binary cross-entropy to measure the loss during training.

Of course, you need to take care of various other parameters in the code but none of this is specific to graph machine learning or to an edge completion task. Rather, it's standard Torch mechanics and can be found as part of any machine learning code or tutorial.

## Testing and predictions

Regarding test and validation there is not much to be said except maybe the fact that one has to use them in the same fashion as the training set. That is, you need to loop over the batches and ensure you look at both positive and negative edges.

There is of course the more complex issue of over-fitting, transductive versus inductive learning and so on. These challenges stem from the fact that a typical graph machine learning task always sees the whole data and one can rarely create a predictive model which can be used outside the observed graph. In the case of edge completion this would of course be a marvelous thing to have: learn the general characteristics of a graph and use it to predict new edges on other graphs. In the context of drug repurposing this would mean that a single model can be used by multiple pharmaceutical companies. In the context of social networks, it would mean that learning friendship characteristics from Facebook can be transposed to, say, Pokec or Twitter and vice versa. In practice this is quite difficult and there are good techniques for node classification task but out-of sample edge predictions is tough. 

With respect to predicting new edges things are straightforward: use negative sampling on the whole graph and attach the model's returned probability to the non-existing edges. Sort the result in decreasing order and take the top N result. Technically this amounts (again) to processing batches, using a sigmoid threshold and truncation to zero or one. All of this is almost identical to the training process and other ML tasks.

Torch and Pyg make it easy to run all things transparently on GPU's and the code takes care of this. In practice, we found that training via batches on CPU is quite fast and that an accuracy of 95% (F1 score) reaches a plateau around 30 epochs. Obviously, GPU processing always is beneficial but the message here is that you can experiment with the code on your laptop and that there is no need to spin up a machine in the cloud to experiment with your own custom implementation or data.   

