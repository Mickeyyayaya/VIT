# ViTCoD Simulator (Sparse)

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
penalty = 3
# root = 'masks/deit/deit_small_lowrank'
# sparse = [0.5]
# embedding = 192
# root = 'masks/levit/LeViT_192_lowrank/0.5'
# root = '/home/sheminghao/shh/ViTCoD/attention_mask'
# root = 'masks/deit_tiny_lowrank'
# sparse = [0.95]
total_preload_linear_cycles = 0
total_preload_clear = 0
total_preload_ffn_cycles = 0
total_linear_PE_cycles = 0
total_ffn_PE_cycles = 0
total_PRE_cycles = 0
total_preload_clear_ffn = 0

log = logging.getLogger()
# TODO:
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

for sram_size in range(10,100*10,100):
    log.info('***' * 10)
    log.info('Mode: Bandwidth {}'.format(sram_size))
    for p in args.sparse:
        # Initialize Q, K, V and attn maps
        # TODO: load the masks of attention and global tokens
        # attn_map_mask = np.load(args.root+'/reodered_info_'+str(p)+'.npy')
        # num_global_tokens = np.load(args.root+'/global_token_info_'+str(p)+'.npy')
        attn_map_mask = np.load(args.root+'/reodered_info_'+str(p)+'.npy')
        num_global_tokens = np.load(args.root+'/global_token_info_'+str(p)+'.npy')
        # attn_map_mask = np.load(args.root+'/reodered_mask_'+str(p)+'.npy')
        # num_global_tokens = np.load(args.root+'/global_token_mask_'+str(p)+'.npy')
        all_Q = np.random.random((attn_map_mask.shape[0], attn_map_mask.shape[1], attn_map_mask.shape[2], args.feature_dim))
        all_K = np.random.random((attn_map_mask.shape[0], attn_map_mask.shape[1], attn_map_mask.shape[2], args.feature_dim))
        all_V = np.random.random((attn_map_mask.shape[0], attn_map_mask.shape[1], attn_map_mask.shape[2], args.feature_dim))
        log.info('Shape: {}'.format(all_V.shape))
        #sram_size = 100*8
        bandwidth = 76.8 * 1024 * 1024 * 1024 * 8 
        my_SRAM = SRAM(sram_size,bandwidth)
        my_PE = PE_array()

        head = all_Q.shape[1]
        # TODO: the compression ratio
        # ratio = 2/3H
        # 
        # embedding = 192
        # if attn_map_mask.shape[1] == 3:
        #     ratio = 2/3
        # elif attn_map_mask.shape[1] == 5:
        #     ratio = 3/5
        # elif attn_map_mask.shape[1] == 6:
        #     ratio = 4/6
        # if p == 'dpt0':
        #     embedding = 192
        # elif p == 'dpt1':
        #     embedding = 288
        # elif p == 'dpt2':
        #     embedding = 384
        
        # PE_width = 64
        # PE_height = 8
        

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
                        if (args.PE_width*args.PE_height) > sram_size: 
                            SDDMM_PE_cycles += penalty
                        SDDMM_PE_cycles += 1
                #print(SDDMM_PE_cycles)
                # ############ K #########
                for _sta_q in range(K.shape[0]): 
                    if _sta_q == 0:
                        for f in range(K.shape[1]):
                            cycles, status = my_SRAM.preload_weight(nums=head*1*args.embedding, bits=8, bandwidth_ratio=1)
                            preload_cycles += cycles
                            if status == 'clear':
                                preload_clear += 1
                
                    for k in range(int((args.embedding* K.shape[1])//int(args.PE_width*args.PE_height/head))):
                        if (args.PE_width*args.PE_height) > sram_size: 
                            SDDMM_PE_cycles += penalty
                        SDDMM_PE_cycles += 1
                
                # ############ V #########
                for _sta_q in range(V.shape[0]): 
                    if _sta_q == 0:
                        for f in range(V.shape[1]):
                            cycles, status = my_SRAM.preload_weight(nums=head*1*args.embedding, bits=8, bandwidth_ratio=1)
                            preload_cycles += cycles
                            if status == 'clear':
                                preload_clear += 1          
                    for v in range(int((args.embedding* V.shape[1])//int(args.PE_width*args.PE_height/head))):
                        if (args.PE_width*args.PE_height) > sram_size: 
                            SDDMM_PE_cycles += penalty
                        SDDMM_PE_cycles += 1

                # store back to DRAM
                # process Q
                for num in range(Q.shape[0]): 
                    if num == 0:
                        cycles, status = my_SRAM.preload_encoder(nums=head*1, bits=8, bandwidth_ratio=1/(head*args.ratio))
                        preload_cycles += cycles
                        if status == 'clear':
                            preload_clear += 1          
                    # ######### Preprocessing 
                    for k in range(math.ceil((head*1* Q.shape[1])//int(args.PE_width*args.PE_height/(head*args.ratio)))):
                        PRE_cycles += 1
                    # ######### Store back 
                    cycles, status = my_SRAM.store_out(nums=1* Q.shape[1], bits=8, bandwidth_ratio=1/(head*args.ratio))
                    preload_cycles += cycles
                    if status == 'clear':
                        preload_clear += 1  

                # process K
                for num in range(K.shape[0]): 
                    if num == 0:
                        cycles, status = my_SRAM.preload_encoder(nums=head*1, bits=8, bandwidth_ratio=1/(head*args.ratio))
                        preload_cycles += cycles
                        if status == 'clear':
                            preload_clear += 1   
                    # ######### Preprocessing 
                    for k in range(math.ceil((head*1* K.shape[1])//int(args.PE_width*args.PE_height/(head*args.ratio)))):
                        PRE_cycles += 1
                    # ######### Store back 
                    cycles, status = my_SRAM.store_out(nums=1* K.shape[1], bits=8, bandwidth_ratio=1/(head*args.ratio))
                    preload_cycles += cycles
                    if status == 'clear':
                        preload_clear += 1  
                # process V
                for num in range(V.shape[0]): 
                    # ######### Store back 
                    cycles, status = my_SRAM.store_out(nums=1* V.shape[1], bits=8, bandwidth_ratio=1/(head))
                    preload_cycles += cycles
                    if status == 'clear':
                        preload_clear += 1  

                # ############# concat of multi-head #######################
                for _tile_attn in range(int(Q.shape[0]*args.embedding*Q.shape[1]// int(args.PE_height*args.PE_width/head))):
                    if (args.PE_width*args.PE_height) > sram_size: 
                        SDDMM_PE_cycles += penalty
                    SDDMM_PE_cycles += 1
                for num in range(Q.shape[0]// int(args.PE_height*args.PE_width/head)):
                    for _tile_attn in range(args.embedding):
                        cycles, status = my_SRAM.preload_weight(nums=Q.shape[1], bits=8, bandwidth_ratio=1/head)  
                        preload_cycles += cycles
                        if status == 'clear':
                            preload_clear += 1   

                total_preload_linear_cycles += preload_cycles
                total_preload_clear += preload_clear
                total_PRE_cycles += PRE_cycles
                total_linear_PE_cycles += SDDMM_PE_cycles
                
                log.info('Preload Clear | cycles: {}'.format(preload_clear))
                log.info('Embedding dataloader | cycles: {}'.format(preload_cycles))
                log.info('Embedding decoder | cycles: {}'.format(PRE_cycles))
                log.info('Embedding calcuation | cycles: {}'.format(SDDMM_PE_cycles))    
                my_SRAM.preload_weight(nums=Q.shape[1], bits=8, bandwidth_ratio=1/head)  
                

                log.info('***' * 4)
                SDDMM_PE_cycles = 0
                preload_cycles = 0
                preload_clear = 0
                # ############# FFN #######################
                for _tile_attn in range(int(Q.shape[0]*args.embedding*args.embedding*4// int(args.PE_height*args.PE_width))):
                    if (args.PE_width*args.PE_height) > sram_size: 
                        SDDMM_PE_cycles += penalty
                    SDDMM_PE_cycles += 1
                for num in range(Q.shape[0]// int(args.PE_height*args.PE_width)):
                    for _tile_attn in range(args.embedding*4):
                        cycles, status = my_SRAM.preload_weight(nums=args.embedding, bits=8, bandwidth_ratio=1)
                        preload_cycles += cycles
                        if status == 'clear':
                            preload_clear += 1 
                # if not 'strided' in p:
                for _tile_attn in range(int(Q.shape[0]*args.embedding*args.embedding*4// int(args.PE_height*args.PE_width))):
                    if (args.PE_width*args.PE_height) > sram_size: 
                        SDDMM_PE_cycles += penalty
                    SDDMM_PE_cycles += 1
                for num in range(Q.shape[0]// int(args.PE_height*args.PE_width)):
                    for _tile_attn in range(args.embedding):
                        cycles, status = my_SRAM.preload_weight(nums=args.embedding*4, bits=8, bandwidth_ratio=1)
                        preload_cycles += cycles
                        if status == 'clear':
                            preload_clear += 1 

                total_preload_ffn_cycles += preload_cycles
                total_ffn_PE_cycles += SDDMM_PE_cycles
                total_preload_clear_ffn += preload_clear

                log.info('Embedding dataloader | cycles: {}'.format(preload_cycles))
                log.info('Embedding calcuation | cycles: {}'.format(SDDMM_PE_cycles))
                log.info('Preload Clear | cycles: {}'.format(preload_clear))
            
        
    log.info('')
    log.info('***' * 10)
    log.info('total linear preprocessing cycles: {}'.format(total_PRE_cycles))
    log.info('total linear preloading cycles: {}'.format(total_preload_linear_cycles))
    log.info('total linear computation cycles: {}'.format(total_linear_PE_cycles))
    log.info('total linear clear cycles: {}'.format(total_preload_clear))
    log.info('total ffn preloading cycles: {}'.format(total_preload_ffn_cycles))
    log.info('total ffn computation cycles: {}'.format(total_ffn_PE_cycles))
    log.info('total ffn clear cycles: {}'.format(total_preload_clear_ffn))

    log.info('')
    linear = max(total_linear_PE_cycles+total_PRE_cycles,total_preload_linear_cycles)
    ffn = max(total_ffn_PE_cycles ,total_preload_ffn_cycles)
    log.info('total linear cycles: {}'.format(linear))
    log.info('total ffn cycles: {}'.format(ffn))
    log.info('total cycles: {}'.format(ffn+linear))
    log.info('***' * 10)
    total_ffn_PE_cycles = 0
    total_preload_linear_cycles = 0
    total_linear_PE_cycles= 0
    total_preload_clear= 0
    total_preload_ffn_cycles= 0
    total_ffn_PE_cycles= 0
    total_preload_clear_ffn = 0