# PRONTO
A Framework for fast {PRO}totyping and benchmarking of S{N}N hardware using {TO}rch-based machine learning dialects.

## Requirements (tested with the following versions)
- numpy 1.26.4
- binary_fractions 1.1.0 
- torch 2.4.0
- snntorch 0.9.1
- tqdm 4.66.5
- opencv-python 4.10.0.84

## How to use PRONTO?
The following two commands can be used to make PRONTO ready for MNIST database.

Before you run the following, make sure you download the MNIST dataset (mnist.npz) and keep in the dataset directory,

- python3 mnist.py
- python3 PRONTO.py
- Update parameters/tb_quantisenc_parameters.vh to change the working directory to the base directory.
- Run Vivado simulation by adding the testbench tb/tb_quantisenc.sv, parameters/parameters.vh, and parameters/tb_quantisenc_parameters.vh.

## Citations
If you find this code useful in your research, please cite our paper:

S. Matinizadeh and A. Das. "An Open-Source and Extensible Framework for Fast Prototyping and Benchmarking of Spiking Neural Network Hardware" International Conference on Field-Programmable Logic and Applications (FPL), 2024

@inproceedings{pronto,

title={An Open-Source and Extensible Framework for Fast Prototyping and Benchmarking of Spiking Neural Network Hardware},

author={Matinizadeh, S. and Das, A.},

booktitle ={IEEE International Conference on Field-Programmable Logic and Applications (FPL)},

year={2024},

publisher={IEEE}

}

## Contact
Ms. Shadi Matinizadeh (sm4884@drexel.edu).
Dr. Anup Das (anup.das@drexel.edu)

## TODO
Convert PRONTO to jupyter notebook.
