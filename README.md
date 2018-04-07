# SimplePythonRNN
This is RNN, LSTM & GRU with Bidirectional only in python & numpy. It is simple and slow but will get the job done :+1:

## Specification
**Type :** Vanilla, LSTM, GRU + Bidrectional

**Weight Initialization :** HE Normal

**Weight Update Policy :** ADAM, ADAGRAD

**Regulization :** Droupout

## Prerequisites
numpy (+ mkl for intel processors. recommend [anaconda](https://www.continuum.io/downloads))


## Example

```python


seq_length = 128
learning_rate = 1e-3
rnn = RNN(
    [
        {'type': 'lstm', 'hidden_size': 256},
        {'type': 'lstm', 'hidden_size': 256},
        {'type': 'lstm', 'hidden_size': 256},
        # {'type': 'lstm', 'hidden_size': vocab_size, 'dropout': 0.5, 'bi': True, 'u_type': 'adagrad'},
    ],
    vocab_size, learning_rate, bi=False)

```

Example RNN is trained by Tolstoy's famous novel "Anna Karenina".


> she timed in anna, and coulderent a government manner, stiva fields, scowling him pertuable to vrinss myself, and quieted it out in any melancholorilly his passing feeling of the intensery which she would forget him, and they his brother's with tcharge upon him in her eyes "for you who this name," the son in amiable of his shohy lots.

Generated string over 65000 iterations.




## API Reference



| Parameter | Description |
| --- | --- |
| type | 'lstm', 'gru', 'vanilla' |
| hidden_size | hiddden dimension size  |
| lr | learning rate |
| u_type | 'adam', 'adagrad' |
| bi | Bidirectional. True, False |


## License
MIT

