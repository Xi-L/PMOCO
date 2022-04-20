from dataclasses import dataclass
import torch
import numpy as np

from MOKProblemDef import get_random_problems

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)

class KPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        self.problems = get_random_problems(batch_size, self.problem_size)
        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        
        # MOKP
        ###################################
        self.items_and_a_dummy = torch.Tensor(np.zeros((self.batch_size, self.problem_size+1, 3)))
        self.items_and_a_dummy[:, :self.problem_size, :] = self.problems
        self.item_data = self.items_and_a_dummy[:, :self.problem_size, :]
        
        if self.problem_size == 50:
            capacity = 12.5
        elif self.problem_size == 100:
            capacity = 25
        elif self.problem_size == 200:
            capacity = 25
        else:
            raise NotImplementedError
        self.capacity = torch.Tensor(np.ones((self.batch_size, self.pomo_size))) * capacity
        
        self.accumulated_value_obj1 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))
        self.accumulated_value_obj2 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))
        
        self.ninf_mask_w_dummy = torch.zeros(self.batch_size, self.pomo_size, self.problem_size+1)
        self.ninf_mask = self.ninf_mask_w_dummy[:, :, :self.problem_size]
        
        self.fit_ninf_mask = None
        self.finished = torch.BoolTensor(np.zeros((self.batch_size, self.pomo_size)))
       

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
       
        # MOKP
        ###################################
        self.items_and_a_dummy = torch.Tensor(np.zeros((self.batch_size, self.problem_size+1, 3)))
        self.items_and_a_dummy[:, :self.problem_size, :] = self.problems
        self.item_data = self.items_and_a_dummy[:, :self.problem_size, :]
        
        if self.problem_size == 50:
            capacity = 12.5
        elif self.problem_size == 100:
            capacity = 25
        elif self.problem_size == 200:
            capacity = 25
        else:
            raise NotImplementedError
        self.capacity = torch.Tensor(np.ones((self.batch_size, self.pomo_size))) * capacity
        
        self.accumulated_value_obj1 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))
        self.accumulated_value_obj2 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))
       
        self.ninf_mask_w_dummy = torch.zeros(self.batch_size, self.pomo_size, self.problem_size+1)
        self.ninf_mask = self.ninf_mask_w_dummy[:, :, :self.problem_size]
        
        self.fit_ninf_mask = None
        self.finished = torch.BoolTensor(np.zeros((self.batch_size, self.pomo_size)))
       
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        self.step_state.capacity = self.capacity
        self.step_state.finished = self.finished

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
       
        # Status
        ####################################
        items_mat = self.items_and_a_dummy[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size+1, 3)
        gathering_index = selected[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, 3)
        selected_item = items_mat.gather(dim=2, index=gathering_index).squeeze(dim=2)
       
        self.accumulated_value_obj1 += selected_item[:, :, 1]
        self.accumulated_value_obj2 += selected_item[:, :, 2]
        self.capacity -= selected_item[:, :, 0]

        batch_idx_mat = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        group_idx_mat = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.ninf_mask_w_dummy[batch_idx_mat, group_idx_mat, selected] = -np.inf

        unfit_bool = (self.capacity[:, :, None] - self.item_data[:, None, :, 0]) < 0
        self.fit_ninf_mask = self.ninf_mask.clone()
        self.fit_ninf_mask[unfit_bool] = -np.inf

        self.finished = (self.fit_ninf_mask == -np.inf).all(dim=2)
        done = self.finished.all()
        self.fit_ninf_mask[self.finished[:, :, None].expand(self.batch_size, self.pomo_size, self.problem_size)] = 0
       
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.fit_ninf_mask
        self.step_state.capacity = self.capacity
        self.step_state.finished = self.finished
        
        reward = None
        if done:
            reward = torch.stack([self.accumulated_value_obj1,self.accumulated_value_obj2],axis = 2)
        else:
            reward = None

        return self.step_state, reward, done

   
