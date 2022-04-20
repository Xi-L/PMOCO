from dataclasses import dataclass
import torch

from MOTSProblemDef_3obj import get_random_problems, augment_xy_data_by_n_fold_3obj

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


class TSPEnv:
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
            if aug_factor <= 512:
                self.batch_size = self.batch_size * aug_factor
                self.problems = augment_xy_data_by_n_fold_3obj(self.problems, aug_factor)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

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
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
      
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 6)
        # shape: (batch, pomo, problem, 6)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 6)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        
        segment_lengths_obj1 = ((ordered_seq[:, :, :, :2]-rolled_seq[:, :, :, :2])**2).sum(3).sqrt()
        segment_lengths_obj2 = ((ordered_seq[:, :, :, 2:4]-rolled_seq[:, :, :, 2:4])**2).sum(3).sqrt()
        segment_lengths_obj3 = ((ordered_seq[:, :, :, 4:]-rolled_seq[:, :, :, 4:])**2).sum(3).sqrt()

        travel_distances_obj1 = segment_lengths_obj1.sum(2)
        travel_distances_obj2 = segment_lengths_obj2.sum(2)
        travel_distances_obj3 = segment_lengths_obj3.sum(2)
    
        travel_distances_vec = torch.stack([travel_distances_obj1,travel_distances_obj2,travel_distances_obj3],axis = 2)
        
        # shape: (batch, pomo)
        return travel_distances_vec

