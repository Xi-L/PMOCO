import torch
import numpy as np
from logging import getLogger

from MOCVRPEnv import CVRPEnv as Env
from MOCVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint_mocvrp-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score_obj1, train_score_obj2, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score_obj1', epoch, train_score_obj1)
            self.result_log.append('train_score_obj2', epoch, train_score_obj2)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            
            # Save Model
            if epoch == self.start_epoch or all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint_mocvrp-{}.pt'.format(self.result_folder, epoch))

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM_obj1 = AverageMeter()
        score_AM_obj2 = AverageMeter()
        
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score_obj1, avg_score_obj2, avg_loss = self._train_one_batch(batch_size)
            score_AM_obj1.update(avg_score_obj1, batch_size)
            score_AM_obj2.update(avg_score_obj2, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg))

        return score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
      
        pref = torch.rand([2])
        pref = pref / torch.sum(pref)
        
        reset_state, _, _ = self.env.reset()
        
        self.model.decoder.assign(pref)
        self.model.pre_forward(reset_state)
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        selected_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            selected_list = torch.cat((selected_list, selected[:, :, None]), dim=2)
            
        # Loss
        ###############################################
        # reward was negative, here we set it to positive to calculate TCH
        reward = - reward
        z = torch.ones(reward.shape).cuda() * 0.0
        tch_reward = pref * (reward - z)      
        tch_reward , _ = tch_reward.max(dim = 2)
        tch_reward = tch_reward
        
        # set back reward and group_reward to negative
        reward = -reward
        tch_reward = -tch_reward
    
        log_prob = prob_list.log().sum(dim=2)
        ## shape = (batch, group)

        tch_advantage = tch_reward - tch_reward.mean(dim=1, keepdim=True)

        tch_loss = -tch_advantage * log_prob # Minus Sign
        # shape = (batch, group)
        loss_mean = tch_loss.mean()
    
        # Score
        ###############################################
        _ , max_idx = tch_reward.max(dim=1)
        max_idx = max_idx.reshape(max_idx.shape[0],1)
        max_reward_obj1 = reward[:,:,0].gather(1, max_idx)
        max_reward_obj2 = reward[:,:,1].gather(1, max_idx)
        
        score_mean_obj1 = - max_reward_obj1.float().mean()
        score_mean_obj2 = - max_reward_obj2.float().mean()

        # Step & Return
        ################################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
            
       
        return score_mean_obj1.item(), score_mean_obj2.item(), loss_mean.item()
