import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p
        self.forward_probabilities = np.empty([])


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables

        # First column in forward probabilities = hot, second = cold 
        # rows = each state in sequence
        # Ex. ForwardP[1,0] = P(X2 = Rainy | Hot), ForwardP[1,1] = P(X2 = Rainy | Cold)

        # initialize forward probabilities here 

        # P(first_state) = prior_probability * emission_probability(hidden state -> observation state)
        self.forward_probabilities = np.zeros([len(input_observation_states), len(self.hidden_states)])
        self.forward_probabilities[0, :] = self.prior_p * self.emission_p[:, self.observation_states_dict[input_observation_states[0]]]
        print(self.emission_p[:, self.observation_states_dict[input_observation_states[0]]])
        # Step 2. Calculate probabilities

        # For each observation state in sequence 
        for t in range(1, len(input_observation_states)):
            # loop through hidden states/emission matrix rows 
            for j in range(len(self.hidden_states)):
                # loop through transition probabilities columns 
                for i in range(len(self.hidden_states)):
                    self.forward_probabilities[t, j] += self.forward_probabilities[t-1, i] * self.transition_p[i, j] * self.emission_p[j, self.observation_states_dict[input_observation_states[t]]]
                
        # Step 3. Return final probability 
        return np.sum(self.forward_probabilities[len(input_observation_states) - 1])


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        # Step 1. Initialize variables
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))
        backtrack_table = np.zeros((len(decode_observation_states), len(self.hidden_states)), dtype=int)
        # calculate initial state probabilities
        viterbi_table[0] = self.prior_p * self.emission_p[:, self.observation_states_dict[decode_observation_states[0]]]

        # Step 2. Calculate Probabilities
        for t in range(1, len(decode_observation_states)):
            for j in range(len(self.hidden_states)):
                # take max probability of prior probability * transition_p row 
                max_prob = np.max(viterbi_table[t - 1] * self.transition_p[:, j])
                # take index of max prob and store in backtrack table 
                prev_state = np.argmax(viterbi_table[t - 1] * self.transition_p[:, j])
                # update your probability table by multiplying max with emission probability 
                viterbi_table[t, j] = max_prob * self.emission_p[j, self.observation_states_dict[decode_observation_states[t]]]
                backtrack_table[t, j] = prev_state

        # Step 3. Traceback 
        best_hidden_state = np.argmax(viterbi_table[-1])
        # backtrack by going backwards in your backtrack table - start with last hidden state (decoded by 0s and 1s)
        best_path[len(decode_observation_states) - 1] = best_hidden_state
        #loop backwards 
        for t in range(len(decode_observation_states)-2, -1, -1):
            best_hidden_state = backtrack_table[t + 1, best_hidden_state]
            best_path[t] = best_hidden_state

        # Step 4. Return best hidden state sequence 
        return [self.hidden_states_dict[state] for state in best_path]


