# Pytorch Geometric Link Predictions

The [graph data science library (GDS) is a Neo4j plugin](https://neo4j.com/product/graph-data-science/) which allows one to apply machine learning on graphs within Neo4j via easy to use procedures playing nice with the existing Cypher query language. Things like node classifications, edge predictions, community detection and more can all be performed inside the database and augment the existing graph with learned characteristics. There are many advantages if you follow this path but it might also not always be sufficient:

- tuning and parametrization is limited
- scalability and performance can be a hurdle
- lack of GPU (TPU) support
- limited set of algorithms
- neural network engineering (assembling layers) is not possible.

There are highly sophisticated graph machine learning (ML) frameworks which can alleviate these obstacles and once the 'learning' has been performed, the predictions can be returned to Neo4j. This means that the ML part is taken outside Neo4j but, in any case, one seldom performs intensive task on a database which potentially block ingestion and serving downstream tasks (website and alike).   

Pytorch Geometric (Pyg) has a whole arsenal of neural network layers and techniques to approach machine learning on graphs (aka graph representation learning, graph machine learning, deep graph learning) and has been used in this repo to learn link patterns, alas known as link or edge predictions. Other frameworks ([Tensorflow Geometric](https://blog.tensorflow.org/2021/11/introducing-tensorflow-gnn.html), [StellarGraph](https://www.stellargraph.io), [DGL](https://www.dgl.ai)...) can give equivalent results and although Pyg is a popular choice, it all depends on your particular context.    

Although Pyg has plenty of examples, this repo contains a few ingredients which you will not find elsewhere:

- how to apply link prediction to a fairly large graph (10M nodes and 30M edges) on a normal device (no GPU, no big data infrastructure)
- how to extract concrete predictions (non-existent edges)
- how it pertains to real-world business cases (drug repurposing e.g.)
- how to create your own Pyg dataset.


More details and context can be found in [this article](https://bratanic-tomaz.medium.com) and is a collaboration of [Tomaz Bratanic (Neo4j)](https://bratanic-tomaz.medium.com) and [Francois Vanderseypen (Orbifold Consulting)](https://graphsandnetworks.com).  

## Setup

Like any Python project you probably should create an environment, something like

    conda create --name py python=3.7

Activate the environment and install the requirements

    conda activate pyg
    pip install -r requirements.txt

If you wish to visualize the training, install `tensorboard`
    
    conda install tensorboard

## Run

Take a look at `run.py` and inspect the parameters:

- how many epochs
- the batch size
- the relative size of test and validation sets
- the maximum amount of predictions to output

and so on.

The code is fairly straightforward to understand and we have tried to comment the unusual bits.

Thereafter, simply execute

    python run.py

in the environment and the training/prediction will start. You can halt the loop at any point (ctrl+c), the model and predictions will be outputted despite the interrupt.

The loss and accuracy are written for each run and can be inspected via

    tensorboard --logdir=runs


## Data

The social graph is a portion of the very popular Pokec network. Pokec is the most popular on-line social network in Slovakia. The popularity of network has not changed even after the coming of Facebook. It has been online for more than 10 years and connects more than 1.6 million people. The dataset contains anonymized data of the whole network.     

The relevant datasets will be automatically [downloaded from SNAP](https://snap.stanford.edu/data/soc-pokec.html), transformed and saved the first time you run the script. A Pyg dataset is created and can be used for other training tasks if you wish.

Part of the data transformation is the creation of a node and edge Pandas dataframe which you can load separately if you wish from the HDF5 `frames.h5` file:

    import pandas as pd
    nodes = pd.read_hdf(hdf_path, "/dfn")
    edges = pd.read_hdf(hdf_path, "/dfe")

The Pyg dataset is automatically loaded if present and is serialized as `pokec.pt` in the `processed` directory. By default you get a `data` directory unless you specify one

    from pokec import Pokec    
    p = Pokec()
        
    # p = Pokec("~/myPygDataDirectory")
