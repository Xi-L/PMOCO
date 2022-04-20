##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config
import os
import sys
import torch
import numpy as np


os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src


from MOCVRPTester import CVRPTester as Tester
from MOCVRProblemDef import get_random_problems
##########################################################################################
import time
import hvwfg


from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
##########################################################################################
# parameters
env_params = {
    'problem_size': 50,
    'pomo_size': 50,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/saved_cvrp50_model',  # directory path of pre-trained model and log files saved.
        'epoch': 200, # epoch version of pre-trained model to laod.
    },
    'test_episodes': 100, 
    'test_batch_size': 100,
    'augmentation_enable': True,
    'aug_factor': 1, #8, 
    'aug_batch_size': 100 
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__cvrp_n50',
        'filename': 'run_log'
    }
}

##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]
##########################################################################################
def main(n_sols = 101):

    timer_start = time.time()
    logger_start = time.time()
    
    if DEBUG_MODE:
        _set_debug_mode()
    
    create_logger(**logger_params)
    _print_config()
    
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    
    copy_all_src(tester.result_folder)
    
    sols = np.zeros([n_sols, 2])

    shared_depot_xy, shared_node_xy, shared_node_demand = get_random_problems(tester_params['test_episodes'], env_params['problem_size'])
    
    for i in range(n_sols):
        pref = torch.zeros(2).cuda()
        pref[0] = 1 - 0.01 * i
        pref[1] = 0.01 * i
        pref = pref / torch.sum(pref)
    
        aug_score = tester.run(shared_depot_xy, shared_node_xy, shared_node_demand,pref)
        sols[i] = np.array(aug_score)
        
    timer_end = time.time()
    
    total_time = timer_end - timer_start
   
    fig = plt.figure()
    
    plt.plot(sols[:,0],sols[:,1], marker = 'o', c = 'C1',ms = 3,  label='PSL-MOCO (Ours)')
    
    plt.legend()
    
    #ref = np.array([15,3])    #20
    ref = np.array([40,3])   #50
    #ref = np.array([60,3])   #100
    
    hv = hvwfg.wfg(sols.astype(float), ref.astype(float))
    hv_ratio =  hv / (ref[0] * ref[1])

    print('Run Time(s): {:.4f}'.format(total_time))
    print('HV Ratio: {:.4f}'.format(hv_ratio))

##########################################################################################
if __name__ == "__main__":
    main()