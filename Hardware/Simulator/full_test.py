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

total_preload_linear_cycles = 0
total_preload_clear = 0
total_preload_ffn_cycles = 0
total_linear_PE_cycles = 0
total_ffn_PE_cycles = 0
total_PRE_cycles = 0
total_preload_clear_ffn = 0
total_preload_cycles = 0
total_SDDMM_PE_cycles = 0
total_SpMM_PE_cycles = 0
total_linear_PE_cycles = 0
total_PRE_cycles = 0
total_input_cycles = 0
total_weight_cycles = 0

for sram_size in range(1000,10000,1000):
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
                
                mask = attn_map_mask[_layer, _head]
                global_tokens = int(num_global_tokens[_layer, _head])
                log.info('global tokens: {}'.format(global_tokens))
                sparser = coo_matrix(1 - mask[:, global_tokens:])
                sparser = np.column_stack((sparser.row, sparser.col))
                if global_tokens == mask.shape[1]:
                    sparse_ratio = 0
                else:
                    sparse_ratio = len(sparser)/(mask[:, global_tokens:].shape[0]*mask[:, global_tokens:].shape[1])
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
                        SDDMM_PE_cycles += math.ceil(int(args.PE_width*args.PE_height/head)/element)
    
                # ############ K #########
                for _sta_q in range(K.shape[0]): 
                    if _sta_q == 0:
                        for f in range(K.shape[1]):
                            cycles, status = my_SRAM.preload_weight(nums=head*1*args.embedding, bits=8, bandwidth_ratio=1)
                            preload_cycles += cycles
                            if status == 'clear':
                                preload_clear += 1
                
                    for k in range(int((args.embedding* K.shape[1])//int(args.PE_width*args.PE_height/head))):
                        SDDMM_PE_cycles += 1
                        SDDMM_PE_cycles += math.ceil(int(args.PE_width*args.PE_height/head)/element)
                
                # ############ V #########
                for _sta_q in range(V.shape[0]): 
                    if _sta_q == 0:
                        for f in range(V.shape[1]):
                            cycles, status = my_SRAM.preload_weight(nums=head*1*args.embedding, bits=8, bandwidth_ratio=1)
                            preload_cycles += cycles
                            if status == 'clear':
                                preload_clear += 1          
                    for v in range(int((args.embedding* V.shape[1])//int(args.PE_width*args.PE_height/head))):
                        SDDMM_PE_cycles += 1
                        SDDMM_PE_cycles += math.ceil(int(args.PE_width*args.PE_height/head)/element)

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
                        PRE_cycles += math.ceil(int(args.PE_width*args.PE_height/head)/element)
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
                    SDDMM_PE_cycles += 1
                    SDDMM_PE_cycles += math.ceil(int(args.PE_width*args.PE_height/head)/element)
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

                log.info('')

                preload_cycles = 0
                weight_cycles = 0
                input_cycles = 0
                ouput_cycles = 0
                PRE_cycles = 0
                SDDMM_PE_cycles = 0
                for _sta_k in range(global_tokens): 
                    # ############ k #########
                    # ######### Load k and decoder weight
                    cycles, status = my_SRAM.preload_K(nums=head*args.ratio*1* K.shape[1], bits=8, bandwidth_ratio=1)  
                    preload_cycles += cycles
                    weight_cycles += cycles
                    if status == 'clear':
                        preload_clear += 1 
                    if _sta_k == 0:
                        cycles, status = my_SRAM.preload_decoder(nums=head*args.ratio*1, bits=8, bandwidth_ratio=1)  
                        preload_cycles += cycles
                        weight_cycles += cycles
                        if status == 'clear':
                            preload_clear += 1 
                    # ######### Preprocessing 
                    for k in range((math.ceil((head*args.ratio*1* K.shape[1])/int(args.PE_width*args.PE_width/head)))):
                        PRE_cycles += 1
                        PRE_cycles += math.ceil(int(args.PE_width*args.PE_height/head)/element)
                    for _sta_q in range(int(Q.shape[0])):
                        if _sta_k == 0:
                        # ############ q #########
                            reload_ratio = 0
                            cycles, status = my_SRAM.preload_Q(nums=head*args.ratio*1* Q.shape[1], bits=8, bandwidth_ratio=1)  
                            preload_cycles += cycles*(1+reload_ratio)
                            input_cycles += cycles*(1+reload_ratio)
                            if status == 'clear':
                                preload_clear += 1 
                            if _sta_q == 0: 
                                cycles, status = my_SRAM.preload_decoder(nums=head*args.ratio*1, bits=8, bandwidth_ratio=1)  
                                preload_cycles += cycles
                                input_cycles += cycles
                                if status == 'clear':
                                    preload_clear += 1 
                            # ######### Preprocessing 
                            for q in range(math.ceil((head*args.ratio*1* Q.shape[1])/int(args.PE_width*args.PE_width/head))):
                                PRE_cycles += 1*(1+reload_ratio)
                                PRE_cycles += math.ceil(int(args.PE_width*args.PE_height/head)/element)
                
                total_PRE_cycles += PRE_cycles
                total_preload_cycles += preload_cycles
                total_input_cycles += input_cycles
                total_weight_cycles += weight_cycles
                log.info('Dense SpMM dataloader | cycles: {}'.format(preload_cycles))
                log.info('Dense SpMM decoder | cycles: {}'.format(PRE_cycles))
                log.info('Input | cycles: {}'.format(input_cycles))
                log.info('weight | cycles: {}'.format(weight_cycles))
                dense_ratio = global_tokens*Q.shape[0]/(len(sparser) + global_tokens*Q.shape[0])
                dense_PE_width = int(args.PE_width*dense_ratio)
                sparse_PE_width = args.PE_width - dense_PE_width
                # ############## dense pattern q*k ##############
                dense_SDDMM_PE_cycles = 0
                for _sta_k in range(global_tokens):
                    for _sta_q in range(math.ceil(Q.shape[0]/dense_PE_width)):
                        for _tile_q in range(math.ceil(Q.shape[1] / (args.PE_width/head))):
                            dense_SDDMM_PE_cycles += 1
                            dense_SDDMM_PE_cycles += math.ceil(int(args.PE_width*args.PE_height/head)/element)
                log.info('Dense SDMM PE caclulation | cycles: {}'.format(dense_SDDMM_PE_cycles))


    log.info('***' * 10)
    log.info('total linear preprocessing cycles: {}'.format(total_PRE_cycles))
    log.info('total linear preloading cycles: {}'.format(total_preload_linear_cycles))
    log.info('total linear computation cycles: {}'.format(total_linear_PE_cycles))
    log.info('total linear clear cycles: {}'.format(total_preload_clear))

    log.info('')
    linear = max(total_linear_PE_cycles+total_PRE_cycles,total_preload_linear_cycles)
    ffn = max(total_ffn_PE_cycles ,total_preload_ffn_cycles)
    log.info('total linear cycles: {}'.format(linear))
    log.info('total cycles: {}'.format(ffn+linear))
    log.info('***' * 10)


    log.info('')
    log.info('***' * 10)

    log.info('total preloading cycles: {}'.format(total_preload_cycles))
    log.info('total processing cycles: {}'.format(total_PRE_cycles))
    log.info('total Computation cycles: {}'.format(total_SDDMM_PE_cycles+total_SpMM_PE_cycles))

    log.info('')
    log.info('***' * 10)
    log.info('Total cycles: {}'.format(max(total_preload_cycles, total_PRE_cycles+total_SDDMM_PE_cycles+total_SpMM_PE_cycles)))




    total_preload_linear_cycles = 0
    total_linear_PE_cycles= 0
    total_preload_clear= 0