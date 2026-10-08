# Challenge Data CFM 2024 - using RNNs

## Files
* `cfm2024.ipynb`: RNN models for the sequential observations: simple RNN, GRU, LSTM. Also WaveNet. The best model parameters are saved in `gru_128_model.pt`
* `cfm2024-feature-engineering.ipynb`: combine the 100-sequence data for each observation into "mean" features. The engineered features are saved in `eng_features.csv`.
* `joint_model.ipynb`: combine sequence inputs and engineered inputs into a single model. 


## Project structure
This is a classification problem with 24 classes, one for each stock. 

We combine different approaches:
* Using RNNs: each observation is a sequence of 100 ticks, each with *n* features (such as price, bid, ask, venue, etc). Thus the instances are a *n*-variate series.
* *n*-variate series can also be predicted with (stacks of) convolutional layers, e.g. with a WaveNet structure.
* Using DNNs **and** RNNs: for each observation, the 100 ticks can be combined in *engineered features*, such as mid price, median price, ratio of venues, etc. 
We build a model where the instances are composed by sequential features *and* engineered features: the sequential input goes through a RNN module, and the output is combined with the engineered input through a (stack of) fully connected layer(s).

## RNNs
### Input
Each instance is a multivariate series. The series consists of 100 sequential order-book entries, where each entry is 12-dimensional: obs_id, venue, action, order_id, side, price, bid, ask, bid_size, ask_size, flux, trade.

obs_id is constant thorugh the series. 

We transform order_id into:
- order_seen_before
- order_previous_count

We keep all the numerical features: price, bid, ask, bid_size, ask_size, flux, trade (bool), order_seen_before (bool), order_previous_count (9).

We keep and one-hot encode the categorial features: action, venue, side (3 + 6 + 2).

In total we have 20 features. So the input data has shape [160800, 100, 20].

Train set: 80% 
Validation set: 20%

### Models
We train in batch sizes of 32, over 50 epochs, using ADAM and cross entropy loss. 
Learning rate is 1e-4.
* Simple RNN model: `simple_model`.
    * nn.RNN with hidden_size = 32
    * nn.linear with output_size = 24

    Validation accuracy: 12%
* GRU model: `gru_model`.
    * nn.GRU with hidden_size = 32
    * nn.linear with output_size = 24

    Validation accuracy: 30%
* LSTM model: `lstm_model`.
    * nn.LSTM with hidden_size = 32
    * nn.linear with output_size = 24

    Validation accuracy: 24%
Since GRU is the best perfomer, we implement
* GRU 64 model: `gru_64_model`.
    * nn.GRU with hidden_size = 64
    * nn.linear with output_size = 24

    Validation accuracy: 39%
* GRU 128 model: `gru_128_model`.
    * nn.GRU with hidden_size = 128
    * nn.linear with output_size = 24

    Validation accuracy: 47%

## WaveNet
### Input
Same as for [RNNs](#input).
### Model
Simplified *WaveNet*. 

*WaveNet*: stacking 1D convolutional layers, doubling the dilation rate (how spread apart each neuron's inputs are) at every layer. The first conv layer sees two time steps at a time, the nex one sees four time steps, etc.

The lower layers learn short-term patterns, while the higher layers learn long-term patterns.

We stack **twice** 1D-convolutional layers each with kernel 2 and dilation that increases: 1,2,4,8.

Accuracy: 20%.

## Feature engineering
Create new features that encode the 100 observations per obs_id, such as mean price, min price, max price, number of total trades, etc.

We implement the following features:

* Price/spread structure:

  * median_price
  * median_mid_price

  * mean_spread
  * median_spread
  * std_spread
  * min_spread
  * max_spread

* Book size:
  * mean_bid_size
  * median_bid_size
  * mean_ask_size
  * median_ask_size

* Imbalance:
  * mean_obi
  * median_obi
  * std_obi

* Event composition:
  * trade_frequency
  * cancel_ratio
  * unique_orders

* Flux:
  * mean_abs_flux: typical event size
  * median_abs_flux
  * mean_flux: net-liquid adding or removing
  * std_flux

* Venues:
  * venue_x_ratio

* Actions:
  * add_ratio
  * delete_ratio
  * update_ratio

## Joint model
### Input
The input features consists of sequential features `X_seq` and engineered features `X_eng`.

Each instance of X_seq is 100x14 dimensional

Each instance of X_eng is 30 dimensional.

X_eng is scaled.

### Model
X_seq goes through a nn.GRU layer with hidden size 128.

The output is combined with X_eng and passed to a nn.linear layer (input_size = 30+14) which outputs the 24 classification logits.

## Improvements and variations

## Deep GRU model
Gru model with `n_layers=2`. 

We reach accuracy of 61% in 60 epochs. This is the best model we found overall.

## Deep GRU joint model
Implementing deep GRU (2 layers) in the joint model.
It performs better than the joint model, but not better than Deep gru on the sequential only inputs.

## Input normalization
Normalization of the sequential data (numerical only) doesn't give a better performance. 

We tried both to normalize all the numerical data, and only the bid_size, ask_size, flux.
The latter gives better results, but not as good as without normalization.

## Deep GRU joint with deep top layer

This is `DeepJoint2GRUModel` in *joint-model.ipynb*.

Instead of a single linear layer to combine the engineered features with the output of the 2-GRU-stack,
we stack two linear layers (with ReLU).

We reach only 27% accuracy in 50 epochs.
