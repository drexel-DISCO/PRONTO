import numpy as np
from senclib import fp2q

from senclib import parameters

'''
# parameters
'''
param           = parameters()
layer_enc_bits  = param.hardware.layer_enc_bits
neuron_enc_bits = param.hardware.neuron_enc_bits
fanin_enc_bits  = param.hardware.fanin_enc_bits

def flush_aer(aer,n=100,all_dump=True):
    #this function writes aer to hardware
    aer_in = aer.reshape(len(aer),-1)   #convert to a 2D array
    aer_hw = np.flip(aer_in,axis=1)[0:n,:]     #flip input. verilog uses [(n-1):0] while numpy uses [0:(n-1)]
    ofname = './input/quantisenc.spikes_input.txt'
    np.savetxt(ofname,aer_hw,delimiter='',fmt='%d')

def flush_weights(weights):
    #this function writes weights to hardware
    wts   = list()   #weights
    addrs = list()   #addresses
    n_layers = len(weights) #number of hardware layers
    for l in range(n_layers):
        #for each layer
        wt_mat = weights[l] #weight matrix
        (in_sz,out_sz) = wt_mat.shape
        for j in range(out_sz):
            for i in range(in_sz):
                wt = wt_mat[i,j]
                addr = address_encoding(layer=l,neuron=j,fanin=i)
                wts.append(process_hex(fp2q(wt),nbits=32))
                addrs.append(addr)
    afname = 'weight/synaptic_address.txt'
    dfname = 'weight/synaptic_weight.txt'

    #write addresses
    with open(afname,'w') as f:
        for addr in addrs:
            f.write(addr + '\n')
    #write data
    with open(dfname,'w') as f:
        for wt in wts:
            f.write(str(wt) + '\n')

def address_encoding(layer=1,neuron=16,fanin=256):
    #address = <--8 bits--><--12 bits--><--12 bits-->
    lhex = process_hex(layer,nbits=layer_enc_bits)
    nhex = process_hex(neuron,nbits=neuron_enc_bits)
    fhex = process_hex(fanin,nbits=fanin_enc_bits)
    return lhex + nhex + fhex

def process_hex(x,nbits=4):
    nhex = nbits / 4    #number of hex bits
    xhex = hex(x)       #hex of x
    xhex_array = xhex.split('x')    #xhex converted into array
    xhex_char = xhex_array[1]       #xhex character
    nhex_char = len(list(xhex_char)) #number of hext characters in the converted value
    extra_hex_characters = int(nhex) - nhex_char
    for i in range(extra_hex_characters):
        xhex_char = '0' + xhex_char
    return xhex_char

def flush_neuron_weights(wts,afname,wfname,layer=0,neuron=0):
    #generate address for a single neuron
    wt_layer  = wts[layer]          #layer's weights
    wt_neuron = wt_layer[:,neuron]  #neurons's fanin weights
    addr = list()   #memory address
    data = list()   #memory data
    for i,wt in enumerate(wt_neuron):
        data.append(process_hex(fp2q(wt),nbits=32))
        addr.append(address_encoding(layer=layer,neuron=neuron,fanin=i))
    #write addresses
    with open(afname,'w') as f:
        for a in addr:
            f.write(a + '\n')
    #write data
    with open(wfname,'w') as f:
        for d in data:
            f.write(str(d) + '\n')
    
    programmed_weights = len(addr)
    return programmed_weights

def flush_layer_weights(wts,afname,wfname,layer=0,neurons=2):
    #generate address for a single neuron
    wt_layer  = wts[layer]          #layer's weights
    addr = list()   #memory address
    data = list()   #memory data

    for neuron in range(neurons):
        wt_neuron = wt_layer[:,neuron]  #neurons's fanin weights
        for i,wt in enumerate(wt_neuron):
            data.append(process_hex(fp2q(wt),nbits=32))
            addr.append(address_encoding(layer=layer,neuron=neuron,fanin=i))
        #write addresses
        with open(afname,'w') as f:
            for a in addr:
                f.write(a + '\n')
        #write data
        with open(wfname,'w') as f:
            for d in data:
                f.write(str(d) + '\n')
    
    programmed_weights = len(addr)
    return programmed_weights

def flush_core_weights(wts,afname,wfname):
    #generate address for a single neuron
    addr = list()   #memory address
    data = list()   #memory data

    n_layers = len(wts) #number of layers
    for layer in range(n_layers):
        wt_layer    = wts[layer]          #layer's weights
        neurons     = wt_layer.shape[1]
        for neuron in range(neurons):
            wt_neuron = wt_layer[:,neuron]  #neurons's fanin weights
            for i,wt in enumerate(wt_neuron):
                processed_wt = fp2q(wt,weights=True)
                #print(wt,processed_wt)
                data.append(process_hex(processed_wt,nbits=32))
                addr.append(address_encoding(layer=layer,neuron=neuron,fanin=i))
            #write addresses
            with open(afname,'w') as f:
                for a in addr:
                    f.write(a + '\n')
            #write data
            with open(wfname,'w') as f:
                for d in data:
                    f.write(str(d) + '\n')
    
    programmed_weights = len(addr)
    return programmed_weights

