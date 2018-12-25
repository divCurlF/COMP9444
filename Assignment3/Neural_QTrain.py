import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 20000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.99  # discount factor
INITIAL_EPSILON = 1.0  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
EPSILON_DECAY_STEPS = 250  # decay period

# Number of hidden nodes in the network
NUM_HIDDEN = 200
MAX_REPLAY_SIZE = 10000
BATCH_SIZE = 32


# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])


# Stores Memory of the agent
class ReplayMemory:

    # Memory stores tuples of (state, action, reward, next_state, done)
    def __init__(self, size):
        self.size = size
        self.memory = []

    def getSize(self):
        return len(self.memory)

    # can modify this for prioritised experience replay.
    def addMemory(self, state, action, reward, next_state, done):
        if len(self.memory) > self.size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def getBatch(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))




# TODO: Define Network Graph
def deep_nn(state_in, state_dim, action_dim, hidden_nodes, name="first"):
    # Define W and b of the first layer
    with tf.variable_scope(name):
        W1 = tf.get_variable(
                "W1" + name,
                shape=[state_dim, hidden_nodes],
                initializer=tf.initializers.truncated_normal(stddev=0.1))
        b1 = tf.get_variable(
                "b1" + name,
                shape=[1, hidden_nodes],
                initializer=tf.constant_initializer(0.0))

        # Define W and b of the second layer
        W2 = tf.get_variable(
                "W2" + name,
                shape=[hidden_nodes, hidden_nodes],
                initializer=tf.initializers.truncated_normal(stddev=0.1))
        b2 = tf.get_variable(
                "b2" + name,
                shape=[1, hidden_nodes],
                initializer=tf.constant_initializer(0.0))

        # Define W and b of the third layer
        W3 = tf.get_variable(
                "W3" + name,
                shape=[hidden_nodes, action_dim],
                initializer=tf.initializers.truncated_normal(stddev=0.1))
        b3 = tf.get_variable(
                "b3" + name,
                shape=[1, action_dim],
                initializer=tf.constant_initializer(0.0))

        # Layer1
        logits_layer1 = tf.matmul(state_in, W1) + b1
        output_layer1 = tf.nn.relu(logits_layer1)

        # Layer2
        logits_layer2 = tf.matmul(output_layer1, W2) + b2
        output_layer2 = tf.nn.relu(logits_layer2)

        # Layer3
        logits_layer3 = tf.matmul(output_layer2, W3) + b3
        output_layer3 = logits_layer3

        return output_layer3


# TODO: Network outputs
q_values = deep_nn(
        state_in,
        STATE_DIM, ACTION_DIM, NUM_HIDDEN, name="q_network")

q_action = tf.reduce_sum(
        tf.multiply(q_values, action_in),
        axis=1,
        name="q_actions")


# target network:
target_values = deep_nn(
        state_in,
        STATE_DIM, ACTION_DIM, NUM_HIDDEN, name="target_network")

loss = tf.reduce_sum(tf.square(q_action - target_in), name="loss")
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss, name="optimizer")

q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_network")
q_target_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES,
        scope="target_network")

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


def update_target_network():
    session.run([target_vars.assign(network_vars) for
                target_vars, network_vars in zip(q_target_vars, q_vars)])


def train_network_replay(exp_replay_mem, q_values, state_in):
    # list of tuples (errors, states, actions, rewards, next_states, done)
    batch = exp_replay_mem.getBatch(BATCH_SIZE)
    states = [item[0] for item in batch]
    actions = [item[1] for item in batch]
    rewards = [item[2] for item in batch]
    next_states = [item[3] for item in batch]
    completions = [item[4] for item in batch]
    targets = []

    # for the case when there is no next state.

    target_q_values = target_values.eval(feed_dict={
        state_in: next_states
    })

    next_q_values = q_values.eval(feed_dict={
        state_in: states
    })

    for i in range(0, len(batch)):
        batch_completed = completions[i]
        if batch_completed:
            target = rewards[i]
        else:
            target = rewards[i] + GAMMA*target_q_values[i][np.argmax(next_q_values[i])]
        targets.append(target)

    session.run([optimizer], feed_dict={
        target_in: targets,
        state_in: states,
        action_in: actions
    })


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


exp_replay_mem = ReplayMemory(MAX_REPLAY_SIZE)
t_network_update = 200
train_count = 0
LAMBDA = 0.001

DONE = False
Do_eps = 5

# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon = FINAL_EPSILON + (INITIAL_EPSILON - FINAL_EPSILON)*(2.718)**(-LAMBDA * train_count)

    # Move through env according to e-greedy policy
    total_reward = 0
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        if not DONE:
            total_reward += reward
            exp_replay_mem.addMemory(state, action, reward, next_state, done)
            train_count += 1
            train_network_replay(exp_replay_mem, q_values, state_in)

            if step % t_network_update == 0:
                update_target_network()

        # Update
        state = next_state
        if done:
            break
    if episode % Do_eps == 0:
        if total_reward == 200:
            DONE = True

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    env.render()
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
              'Average Reward:', ave_reward)

env.close()



