import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def softmax_grad(x):
    soft = softmax(x)
    s = soft.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


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
        p_as = self.action_probability(state)
        dirac = list(map(lambda i: i == action, range(self.num_actions)))
        state_t = np.mat(state).transpose()
        grad_softmax = np.mat(dirac - p_as)
        dW = np.matmul(state_t, grad_softmax)
        db = grad_softmax

        return dW, db

    def update_weights(self, dW, db):
        """
        Updates weights using simple gradient ascent
        :param dW: gradients w.r.t W
        :param db: gradients w.r.t b
        """
        self.W += self.lr * dW
        self.b += self.lr * db
