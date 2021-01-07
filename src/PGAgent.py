import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def softmax_grad(x):
    len_x = len(x)
    jacobian_m = np.diag(x)
    for i in range(len_x):
        for j in range(len_x):
            delta_i_eql_j = 1 if i == j else 0
            jacobian_m[i][j] = x[i] * (delta_i_eql_j - x[i])
    return jacobian_m


class PolicyGradientAgent(object):
    def __init__(self, env, lr=0.1):
        self.num_actions = env.action_space.n
        self.num_features = env.observation_space.shape[0]
        self.W = np.zeros((self.num_features, self.num_actions))
        self.b = np.zeros((self.num_actions))
        self.lr = lr

    def get_inner_softmax_fn(self, state):
        inner_softmax_fn = np.matmul(state, self.W) + self.b
        return inner_softmax_fn

    def action_probability(self, state):
        inner_softmax_fn = self.get_inner_softmax_fn(state)
        p_as = softmax(inner_softmax_fn)
        return p_as

    def get_action(self, state):
        """
        Selects a random action according to p(a|s)
        :param state: environment state
        :return: action
        """
        probs = self.action_probability(state)
        return np.random.choice(self.num_actions, p=probs)

    def grad_log_prob(self, state, action):
        inner_softmax_fn = self.get_inner_softmax_fn(state)
        p_as = self.action_probability(state)
        p_as_grad = softmax_grad(inner_softmax_fn)
        dW_p_as = -np.matmul(state, p_as_grad) / p_as
        db_p_as = -np.matmul(np.ones_like(state), p_as_grad) / p_as

        return np.array([dW_p_as[action], db_p_as[action]])

    def update_weights(self, dW, db):
        """
        Updates weights using simple gradient ascent
        :param dW: gradients w.r.t W
        :param db: gradients w.r.t b
        """
        self.W += self.lr * dW
        self.b += self.lr * db
