# Import routines

import numpy as np
import math
import random
import itertools

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

#Utlity function to update day and time once an action has been executed
def update_day_time(state, time):
        current_time = state[1]
        current_day = state[2]
        
        time = int(time)
        current_time += time
        if(current_time >= t):
            current_day += 1
            if(current_day >= d):
                current_day = current_day % d
            current_time = current_time % t
        
        state = list(state)
        state[1] = current_time
        state[2] = current_day
        return tuple(state)


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0,0)] + list(itertools.permutations([i for i in range(m)],2))
        self.state_space = [(x,y,z) for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        month_vec = [0 if x!=state[0] else 1 for x in range(m)]
        time_vec = [0 if x!=state[1] else 1 for x in range(t)]
        day_vec = [0 if x!=state[2] else 1 for x in range(d)]
        state_encod = month_vec + time_vec + day_vec
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. 
        This method converts a given state-action pair into a vector format. 
        Hint: The vector is of size m + t + d + m + m."""
        #state_encod = [0 for _ in range(m+t+d+m+m)]
        month_vec = [0 if x!=state[0] else 1 for x in range(m)]
        time_vec = [0 if x!=state[1] else 1 for x in range(t)]
        day_vec = [0 if x!=state[2] else 1 for x in range(d)]
        source = [0 if x!=action[0] else 1 for x in range(m)]
        destination = [0 if x!=action[1] else 1 for x in range(m)]
        state_encod = month_vec + time_vec + day_vec + source + destination
        return state_encod
        
    


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        else:  #location=4
            requests = np.random.poisson(8)
            
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
                
        actions.append([0,0])
        possible_actions_index = possible_actions_index + [0]
        return possible_actions_index,actions   
   
            

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        #Case when the driver is offline
        if(action[0]==0 and action[1]==0):
            reward = -C
        else:
            #Time from current location to pickup point
            curr_loc=state[0]
            pickup_loc=action[0]
            curr_time = state[1]
            curr_day = state[2]
            
            Time_i_p = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            # update current day-time for the pickup location
            current_state = state
            updated_state = update_day_time(current_state, Time_i_p)
            curr_loc=action[0]
            drop_loc=action[1]
            curr_time=updated_state[1]
            curr_day=updated_state[2]
            
            Time_p_q = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            reward = (R * Time_p_q) -(C * (Time_p_q + Time_i_p))
        return reward



    #step function
    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        current_state = state
        if(action[0]==0 and action[1]==0):
            #Increase time by 1 hour
            next_state = update_day_time(current_state, 1)
            time_to_trip = 1
        else:
            #Time from current location to pickup point
            curr_loc=state[0]
            pickup_loc=action[0]
            curr_time = state[1]
            curr_day = state[2]
            Time_i_p = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            # update current day-time for the pickup location
            next_state = update_day_time(current_state, Time_i_p)
            # next update after completing journey from p-q
            curr_loc=action[0]
            drop_loc=action[1]
            curr_time=next_state[1]
            curr_day=next_state[2]
            Time_p_q = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            next_state = update_day_time(current_state, Time_p_q)
            time_to_trip = Time_i_p + Time_p_q
        return next_state, time_to_trip




    def reset(self):
        return self.action_space, self.state_space, self.state_init
