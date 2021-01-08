import numpy as np
import matplotlib.pyplot as plt
from src.PGAgent import PolicyGradientAgent
import gym
import argparse
from datetime import datetime

parser = argparse.ArgumentParser("")
parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
parser.add_argument("--N", help="Number of updates", type=int, default=2000)
parser.add_argument("--b", help="batch size", type=int, default=1)
parser.add_argument("--seed", help="random seed", type=int, default=1)
parser.add_argument("--RTG", help="reward to go", dest="RTG", action="store_true")
parser.add_argument(
    "--baseline", help="use baseline", dest="baseline", action="store_true"
)


args = parser.parse_args()
np.random.seed(args.seed)

env = gym.make("CartPole-v1")

print(
    "Enviourment: CartPole-v1 \nNumber of actions: ",
    env.action_space.n,
    "\nDimension of state space: ",
    np.prod(env.observation_space.shape),
)


def run_episode(env, agent, reward_to_go=False, baseline=0.0):
    state = env.reset()
    rewards = []
    dW_arr = []
    db_arr = []
    rewards = []
    terminal = False
    counter = 0
    while not terminal:
        counter += 1
        action = agent.get_action(state)
        state, reward, terminal, _ = env.step(action)
        rewards.append(reward)

        grad_log_W, grad_log_B = agent.grad_log_prob(state, action)
        coefficient = sum(rewards) if reward_to_go else 1
        dW_arr.append(coefficient * grad_log_W)
        db_arr.append(coefficient * grad_log_B)

    rewards_sum = sum(rewards)
    dW = np.sum(dW_arr, 0) * (rewards_sum - baseline)
    db = np.sum(db_arr, 0) * (rewards_sum - baseline)

    return dW, db, rewards_sum, counter


def train(env, agent, args):
    rewards = []
    num_of_episodes_for_baseline = 10
    for i in range(args.N):
        dW = np.zeros_like(agent.W)
        db = np.zeros_like(agent.b)
        for j in range(args.b):
            baseline = 0
            if args.baseline and len(rewards) >= num_of_episodes_for_baseline:
                baseline = np.average(rewards[-num_of_episodes_for_baseline:])
            episode_dW, episode_db, r, counter = run_episode(
                env,
                agent,
                args.RTG,
                baseline,
            )
            rewards.append(r)
            dW -= episode_dW / args.b
            db -= episode_db.flatten() / args.b

        agent.update_weights(dW, db)

        if i % 100 == 25:
            temp = np.array(rewards[i - 25 : i])
            dateTimeObj = datetime.now()
            timestampStr = dateTimeObj.strftime("%H:%M:%S")
            print(
                "{}: [{}-{}] reward {:.1f}{}{:.1f}".format(
                    timestampStr,
                    i - 25,
                    i,
                    np.mean(temp),
                    u"\u00B1",
                    np.std(temp) / np.sqrt(25),
                )
            )
    return agent, rewards


def test(env, agent):
    rewards = []
    print("_________________________")
    print("Running 500 test episodes....")
    for i in range(500):
        _, _, r, counter = run_episode(env, agent)
        rewards.append(r)
    rewards = np.array(rewards)
    print(
        "Test reward {:.1f}{}{:.1f}".format(
            np.mean(rewards), u"\u00B1", np.std(rewards) / np.sqrt(500.0)
        )
    )
    return agent, rewards


agent = PolicyGradientAgent(env)
agent, rewards = train(env, agent, args)
print("Average training rewards: ", np.mean(np.array(rewards)))
test(env, agent)
plt.plot(np.cumsum(rewards))
plt.show()
