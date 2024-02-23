import pytest
from hmm import HiddenMarkovModel
import numpy as np

def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    mini_input = np.load('./data/mini_weather_sequences.npz')

    HMM_obj = HiddenMarkovModel(observation_states=mini_hmm['observation_states'], hidden_states=mini_hmm['hidden_states'], prior_p=mini_hmm['prior_p'], transition_p=mini_hmm['transition_p'], emission_p=mini_hmm['emission_p'])

    input_observation_sequence = mini_input['observation_state_sequence']
    best_hidden_state_sequence = mini_input['best_hidden_state_sequence']

    # run forward algorithm
    forward_p = HMM_obj.forward(input_observation_sequence)

    # assert that your starting probabilities are correct - prior_p * emission_p
    # P(Sunny) = P(Sunny | Hot) + P(Sunny | Cold) - sunny bc it is first observed state in sequence. 
    p_sunny_g_hot = 0.6 * 0.65
    p_sunny_g_cold = 0.4 * 0.15

    assert ( HMM_obj.forward_probabilities[0,0] == p_sunny_g_hot )
    assert ( HMM_obj.forward_probabilities[0,1] == p_sunny_g_cold )

    # assert that HMM forward probability is correct - calculated this by hand
    assert ( round(forward_p, 5) == 0.03506 )
    print(HMM_obj.hidden_states_dict)

    # test Viterbi algorithm output here 
    assert( list(best_hidden_state_sequence) == HMM_obj.viterbi(input_observation_sequence) ) 
    assert( len(best_hidden_state_sequence) == len(HMM_obj.viterbi(input_observation_sequence)) )

def test_edge_case_1():

    # Edge Case 1a: what if observed state doesn't exist within model? (ex. snow)

    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    HMM_obj = HiddenMarkovModel(observation_states=mini_hmm['observation_states'], hidden_states=mini_hmm['hidden_states'], prior_p=mini_hmm['prior_p'], transition_p=mini_hmm['transition_p'], emission_p=mini_hmm['emission_p'])

    EC1_obs_seq = ['snow', 'rainy']
    with pytest.raises(KeyError):
        HMM_obj.forward(EC1_obs_seq)

    # Edge Case 1b: what if observed state probability was 0? (ex. given that it is hot, P(snow) = 0). 
    # create your own HMM object. 
    observation_states = np.array(['sunny', 'rainy', 'snow'])
    hidden_states = ["hot", "cold"]

    prior_p = [0.6, 0.4]
    transition_p = np.array([[0.55, 0.45], [0.3 , 0.7]])
    emission_p = np.array([[0.65, 0.35, 0], [0.1, 0.2, 0.7]])

    p_snow_g_hot = 0
    p_snow_g_cold = 0.7 * 0.4

    HMM_EC_1 = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)
    forward_probability = HMM_EC_1.forward(EC1_obs_seq)

    # Assert that P(snowy | hot) = 0, P(snowy | cold) = 0.7
    assert ( HMM_EC_1.forward_probabilities[0,0] == p_snow_g_hot )
    assert ( HMM_EC_1.forward_probabilities[0,1] == p_snow_g_cold )

    # Assert that forward probability is calculated correctly. 

    # So P( rainy | snowy(prev. observed state)) forward_p[1,0] = hot -> hot -> rainy + cold -> hot -> rainy
    # hot -> hot -> rainy = 0, so forward_p[1,0] = cold -> hot -> rainy
    assert ( round(HMM_EC_1.forward_probabilities[1,0], 4) == round(0.28 * 0.3 * 0.35, 4) )

    # Assert forward_p[1,1] = hot -> cold -> rainy and cold -> cold -> rainy, but hot = 0, so forward_p[1,1] = cold -> cold -> rainy
    assert ( round(HMM_EC_1.forward_probabilities[1,1], 4) ==  round(0.28 * 0.7 * 0.2, 4) )

    # Finally, test viterbi algorithm for test case - because the likelihood of snow given hot is 0, your best hidden state sequence is ['cold', 'cold']. 
    assert( ['cold', 'cold'] == HMM_EC_1.viterbi(EC1_obs_seq) )

def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    HMM_obj = HiddenMarkovModel(observation_states=full_hmm['observation_states'], hidden_states=full_hmm['hidden_states'], prior_p=full_hmm['prior_p'], transition_p=full_hmm['transition_p'], emission_p=full_hmm['emission_p'])

    input_observation_sequence = full_input['observation_state_sequence']
    best_hidden_state_sequence = full_input['best_hidden_state_sequence']

    # test Viterbi algorithm output here 
    assert( list(best_hidden_state_sequence) == HMM_obj.viterbi(input_observation_sequence) ) 













