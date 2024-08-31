import math

def q2fp(binstr,integer_precision=2,decimal_precision=4):
    N = 1 + integer_precision + decimal_precision
    Q = decimal_precision
    sign = 1
    while len(binstr)<N:
        binstr = '0'+binstr

    if binstr[0] == '1':
        sign = -1
        negV = -1 * (int(''.join('1' if x == '0' else '0' for x in binstr), 2) + 1)
        negV = 0 - negV
        binstr = bin(negV).replace('0b','')
        while len(binstr) < N:
            binstr = '0'+binstr
    
    bin_str = [int(a) for a in [*binstr]]
    dec_str = bin_str[0:-Q]
    frac_str= bin_str[-Q:]
    intV = 0
    for i,x in enumerate(frac_str):
        intV += x / (2 ** (i+1))

    for i,x in enumerate(dec_str):
        intV += x * (2 ** (len(dec_str) - 1 - i))

    intV *= sign
    
    return intV

