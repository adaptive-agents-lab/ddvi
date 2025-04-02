class AbstractModel:

    # gets a matrix with rows [s_i, a_i, r_i, s_i+1] and updates the model
    def update(self, sars_vector):
        pass

    # sample next_state for a single s,a pair
    def sample_next_state(self, state, action):
        pass

    # sample reward for a single s,a pair
    def sample_reward(self, state, action):
        pass

    # Return full P_hat matrix
    def get_P_hat(self):
        pass

    # Return full r_hat matrix
    def get_r_hat(self):
        pass

    