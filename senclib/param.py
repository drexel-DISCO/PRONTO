#param.py

class parameters():
    def __init__(self):
        self.lif        = lif_params()
        self.mnist      = mnist_params()
        self.hardware   = hardware_params()
        self.model      = model_params()

class lif_params():
    def __init__(self):
        self.vth = 1.1
        self.neuron_c = 1 
        self.neuron_r = 5 
        self.neuron_decay_rate = 1 / (self.neuron_r * self.neuron_c)
        self.neuron_grow_rate = 1 / self.neuron_c
        self.vrest = 0 
        self.reset_mechanism = 2 
        self.refractory_period = 0 
        self.beta = 0.80  #still investigating what this parameter is

class mnist_params():
    def __init__(self):
        self.seed = 40
        self.xdim = 16
        self.ydim = 16
        self.time_samples_per_pixel = 200
        self.max_spike_rate = 100
        self.pad_samples = 10

class hardware_params():
    def __init__(self):
        #quantisenc parameters
        self.integer_precision  = 3
        self.decimal_precision  = 4
        self.data_width         = 32
        self.layer_enc_bits     = 8
        self.neuron_enc_bits    = 12
        self.fanin_enc_bits     = 12

class model_params():
    def __init__(self):
        #network parameters
        self.input_neurons = 256
        self.hidden_layers = 1
        self.hidden_layer_neurons = 16
        self.output_neurons = 10
