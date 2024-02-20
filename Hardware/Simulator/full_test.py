# ViTCoD Simulator (parallel)

import numpy as np
from torch import embedding
from SRAM_m import SRAM
from PE import PE_array
from scipy.sparse import coo_matrix
import logging
import os
import math
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('ViTCoD attn similation script', add_help=False)
    parser.add_argument('--root', default='masks/deit_tiny_lowrank', type=str)
    parser.add_argument('--sparse', type=float, default=[0.95], nargs='+', help='the sparsity of the model')
    parser.add_argument('--feature_dim', default=64, type=int, help='the feature dimension of Q/K/V')
    parser.add_argument('--embedding', default=192, type=int, help='the embedding dimension')
    parser.add_argument('--ratio', default=2/3, type=float, help='the compression ratio of encoder/decoder')
    parser.add_argument('--PE_width', default=64, type=int)
    parser.add_argument('--PE_height', default=8, type=int)
    return parser

parser = argparse.ArgumentParser('ViTCoD attn similation script', parents=[get_args_parser()])
args = parser.parse_args()

log = logging.getLogger()

log_path = os.path.join(args.root, 'vitcod_atten_ffn.txt')
handlers = [logging.FileHandler(log_path, mode='a+'),
            logging.StreamHandler()]
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    handlers=handlers)
len_gb = 0
len_sparse = 0


total_preload_cycle = 0

for sram_size in range(10,100*10,100):
    log.info('***' * 10)
    log.info('Mode: Bandwidth {}'.format(sram_size))
    for p in args.sparse:
        attn_map_mask = np.load(args.root+'/reodered_info_'+str(p)+'.npy')
        num_global_tokens = np.load(args.root+'/global_token_info_'+str(p)+'.npy')
        all_Q = np.random.random((attn_map_mask.shape[0], attn_map_mask.shape[1], attn_map_mask.shape[2], args.feature_dim))
        all_K = np.random.random((attn_map_mask.shape[0], attn_map_mask.shape[1], attn_map_mask.shape[2], args.feature_dim))
        all_V = np.random.random((attn_map_mask.shape[0], attn_map_mask.shape[1], attn_map_mask.shape[2], args.feature_dim))
        log.info('Shape: {}'.format(all_V.shape))
        
        # TODO: Setting the size
        #sram_size = 100*8
        bandwidth = 76.8 * 1024 * 1024 * 1024 * 8 
        freq = 500*1e6
        element = math.ceil(bandwidth/(freq*8))
        my_SRAM = SRAM(sram_size,bandwidth)
        my_PE = PE_array()
        head = all_Q.shape[1]
        for _layer in range(all_Q.shape[0]//all_Q.shape[0]):
            
            for _head in range(head//head):
                #print(_head)
                Q = all_Q[_layer, _head]
                K = all_K[_layer, _head]
                V = all_V[_layer, _head]
                log.info('***' * 10)
                log.info('Layer: {}; Head: {}'.format(_layer, _head))
                # print('***' * 10)
                
                # ############## embedding ##############
                # K-stationary (Why? Because the number of gloal tokens vary a lot --> Score stationary is not best fit)
                preload_cycles = 0
                preload_clear = 0
                cycles = 0
                status = 'none'
                PRE_cycles = 0
                SDDMM_PE_cycles = 0
                # ############ Q #########
                for _sta_q in range(Q.shape[0]): 
                    #print(_sta_q)
                    if _sta_q == 0:
                        for f in range(Q.shape[1]):
                            #print(head*1*args.embedding)
                            cycles, status = my_SRAM.preload_weight(nums=head*1*args.embedding, bits=8, bandwidth_ratio=1)
                            preload_cycles += cycles
                            if status == 'clear':
                                preload_clear += 1
                    for k in range(int((args.embedding* Q.shape[1])//int(args.PE_width*args.PE_height/head))):
                        SDDMM_PE_cycles += 1
                #print(SDDMM_PE_cycles)
