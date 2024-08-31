import math
import numpy as np
from binary_fractions import Binary
#from quantisenc_parameters import *
from senclib  import parameters

def TwosComplement(str):
    n = len(str)

    # Traverse the string to get first
    # '1' from the last of string
    i = n - 1
    while(i >= 0):
        if (str[i] == '1'):
            break

        i -= 1

    # If there exists no '1' concatenate 1
    # at the starting of string
    if (i == -1):
        return '1'+str

    # Continue traversal after the
    # position of first '1'
    k = i - 1
    while(k >= 0):

        # Just flip the values
        if (str[k] == '1'):
            str = list(str)
            str[k] = '0'
            str = ''.join(str)
        else:
            str = list(str)
            str[k] = '1'
            str = ''.join(str)

        k -= 1

    # return the modified string
    return str

def bin2fp(bin_integer,bin_fraction):
    v = 0
    for i,b in enumerate(bin_integer):
        v += b * (2 ** i)

    for i,b in enumerate(bin_fraction):
        v += b * 1 / (2 ** (i+1))

    return v

def max_fp_positive(n=1,q=4):
    bin_integer  = [1 for i in range(n)]
    bin_fraction = [1 for i in range(q)]
    return bin2fp(bin_integer,bin_fraction)

def min_fp_positive(n=1,q=4):
    bin_integer  = [0 for i in range(n)]
    bin_fraction = [0 for i in range(q)]
    bin_fraction[-1] = 1
    return bin2fp(bin_integer,bin_fraction)


def fp2q(fp,weights=False):
    #A quantized representation is of the form n.q, where n is the number of precision bits and
    #q is the number of quantized bits. 
    #keep record if the number is negative and process it as a positive number
    param = parameters()
    n_fraction  = param.hardware.decimal_precision
    n_integer   = param.hardware.integer_precision
    if weights:
        n_integer   = 0
        fp_pos_max  = max_fp_positive(n=n_integer,q=n_fraction)
        fp_pos_min  = min_fp_positive(n=n_integer,q=n_fraction)
        #print(fp_pos_max,fp_pos_min)
        fp_clip = np.clip(abs(fp),fp_pos_min,fp_pos_max)
        if fp < 0:  #
            fp = 0 - fp_clip
        else:
            fp = fp_clip
    #print('FP = ',fp)
    debug       = False
    negative    = False
    if fp < 0:
        fp = -fp
        negative = True

    #extract fraction and integer part
    (frac,dec)  = math.modf(fp)
    dec         = int(dec)
    if debug:
        print('Is the number negative: ',negative)
        print('Decimal component: ',dec, ' of type ',type(dec))
        print('Fraction component: ',frac, ' of type ',type(frac))
    
    #convert to binary
    #process the fraction part here
    ref_frac_str    = ['0' for i in range(n_fraction)]
    if frac == 0:       #if frac == 0, then replace it with a very small number
        frac = 0.0001
    frac        = f"Binary({frac}) = {Binary(frac)}".split('=')[1].split('b')[1].split('.')[1]
    frac_list   = list(frac)

    if len(frac_list) <= n_fraction:
        frac_list = frac_list + ref_frac_str
    frac_part   = frac_list[0:n_fraction]
    if debug:
        print('Converted fraction bits: ',frac)
        print('Fraction bits = ',frac_list)
        print('Truncated fraction bits = ',frac_part)

    #process the integer part here
    ref_dec_str = ['0' for i in range(n_integer)]
    if (dec == 0) or (n_integer == 0):
        dec_list = ref_dec_str
    else:
        dec         = f"Binary({dec}) = {Binary(dec)}".split('=')[1].split('b')[1]
        dec_list    = ref_dec_str + list(dec)

    dec_part    = dec_list[-n_integer:]
    if debug:
        print('Converted decimal bits: ',dec)
        print('Decimal bits = ',dec_list)
        print('Truncated decimal bits = ',dec_part)

    bin_str         = dec_part + frac_part
    if debug:
        print('Binary string with given config: ',bin_str)
    bin_str.insert(0,'0')
    if debug:
        print('Binary string adter sign addition: ',bin_str)
    
    bin_str = ''.join(bin_str)

    if debug:
        print('Absolute binary string: ',bin_str)
    #2's complement if negative
    if negative:
        bin_str = TwosComplement(bin_str)

    decimal_bin_str = int(bin_str,2)
    if debug:
        print('Signed binary string: ',bin_str)
        print('Decimal value: ',decimal_bin_str)

    return decimal_bin_str
