import numpy as np
import pickle
import argparse

from senclib import parameters
from senclib import fp2q
from senclib import flush_core_weights
from senclib import SearchReplaceStr
'''
###############################################################
# instantiate parameters
###############################################################
'''
param                   = parameters()
input_neurons           = param.model.input_neurons
hidden_layers           = param.model.hidden_layers
hidden_layer_neurons    = param.model.hidden_layer_neurons
output_neurons          = param.model.output_neurons
integer_precision       = param.hardware.integer_precision
decimal_precision       = param.hardware.decimal_precision
layer_enc_bits          = param.hardware.layer_enc_bits
neuron_enc_bits         = param.hardware.neuron_enc_bits
fanin_enc_bits          = param.hardware.fanin_enc_bits
vth                     = param.lif.vth
decay_rate              = param.lif.neuron_decay_rate
grow_rate               = param.lif.neuron_grow_rate
vrest                   = param.lif.vrest
reset_mechanism         = param.lif.reset_mechanism
refractory_period       = param.lif.refractory_period
'''
###############################################################
'''

'''
###############################################################
# process arguments
###############################################################
'''
#add arguments
parser = argparse.ArgumentParser()
parser.add_argument('-layer_to_monitor','--layer_to_monitor',default=0)
parser.add_argument('-neuron_to_monitor','--neuron_to_monitor',default=0)
#parse arguments
args        = vars(parser.parse_args())
layer_to_monitor = int(args['layer_to_monitor'])
neuron_to_monitor = int(args['neuron_to_monitor'])
'''
###############################################################
'''

'''
###############################################################
# save model weight
###############################################################
'''
print('[info] writing weights')
mfname      = './output/torch_out.pkl'
afname      = './weight/quantisenc.synaptic_address.txt'
wfname      = './weight/quantisenc.synaptic_weight.txt'

mnist_data  = pickle.load(open(mfname,'rb'))
wts         = mnist_data['weights']
sim_cnt     = mnist_data['n_timesteps']
vmems       = mnist_data['vmems']
spikes      = mnist_data['spikes']
wts_cnt     = flush_core_weights(wts,afname,wfname)

vmem_ref    = vmems[layer_to_monitor][:,neuron_to_monitor]
spike_ref   = spikes[-1].reshape(sim_cnt,-1)
'''
###############################################################
'''
#exit()

'''
###############################################################
# write testbench and design parameters
###############################################################
'''
print('[info] writing testbench parameters')
tbfname = './parameters/tb_quantisenc_parameters.vh'  #testbench parameter name
dfname  = './parameters/parameters.vh'             #design parameters

search_str =  'parameter VTH'
replace_str = 'parameter VTH = '+str(fp2q(vth))+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter DECAY_RATE'
replace_str = 'parameter DECAY_RATE = '+str(fp2q(decay_rate))+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter GROW_RATE'
replace_str = 'parameter GROW_RATE = '+str(fp2q(grow_rate))+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter VREST'
replace_str = 'parameter VREST = '+str(vrest)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter RESET_MECHANISM'
replace_str = 'parameter RESET_MECHANISM = '+str(reset_mechanism)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter REFRACTORY_PERIOD'
replace_str = 'parameter REFRACTORY_PERIOD = '+str(refractory_period)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter LAYER_TO_MONITOR'
replace_str = 'parameter LAYER_TO_MONITOR = '+str(layer_to_monitor)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter NEURON_TO_MONITOR'
replace_str = 'parameter NEURON_TO_MONITOR = '+str(neuron_to_monitor)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter INPUT_NEURONS'
replace_str = 'parameter INPUT_NEURONS = '+str(input_neurons)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=dfname)

search_str =  'parameter OUTPUT_NEURONS'
replace_str = 'parameter OUTPUT_NEURONS = '+str(output_neurons)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=dfname)

search_str =  'parameter HIDDEN_LAYERS'
replace_str = 'parameter HIDDEN_LAYERS = '+str(hidden_layers)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=dfname)

search_str =  'parameter HIDDEN_LAYER_NEURONS'
replace_str = 'parameter HIDDEN_LAYER_NEURONS = '+str(hidden_layer_neurons)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=dfname)

search_str =  'parameter INTEGER_PRECISION'
replace_str = 'parameter INTEGER_PRECISION = '+str(integer_precision)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=dfname)

search_str =  'parameter DECIMAL_PRECISION'
replace_str = 'parameter DECIMAL_PRECISION = '+str(decimal_precision)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=dfname)

search_str =  'parameter LAYER_ENC_BITS'
replace_str = 'parameter LAYER_ENC_BITS = '+str(layer_enc_bits)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=dfname)

search_str =  'parameter NEURON_ENC_BITS'
replace_str = 'parameter NEURON_ENC_BITS = '+str(neuron_enc_bits)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=dfname)

search_str =  'parameter FANIN_ENC_BITS'
replace_str = 'parameter FANIN_ENC_BITS = '+str(fanin_enc_bits)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=dfname)

search_str =  'parameter SIM_CNT'
replace_str = 'parameter SIM_CNT = '+str(sim_cnt)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter WTS_CNT'
replace_str = 'parameter WTS_CNT = '+str(wts_cnt)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)
'''
###############################################################
'''

'''
###############################################################
# write testbench parameters
###############################################################
'''
print('[info] creating references')
vfname = 'ref/quantisenc.vmem.ref.txt'
np.savetxt(vfname,vmem_ref,fmt='%f')

sfname = 'ref/quantisenc.spikes.ref.txt'
np.savetxt(sfname,spike_ref,fmt='%d',delimiter='')
