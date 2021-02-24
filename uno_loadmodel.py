''' An example of loading a pre-trained dqn model on Leduc Hold'em
'''
import tensorflow as tf
import os

import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament

# Make environment
env = rlcard.make('uno', config={'seed': 0})

# Set a global seed
set_global_seed(0)

# Load pretrained model
graph = tf.Graph()
sess = tf.Session(graph=graph)

with graph.as_default():
    dqn_agents = []
    for i in range(env.player_num):
        agent = DQNAgent(sess,
                    scope='dqn' + str(i),
                    action_num=env.action_num,
                    state_shape=env.state_shape,
                    mlp_layers=[512,512])
        dqn_agents.append(agent)

# We have a pretrained model here. Change the path for your model.
check_point_path = os.path.join(rlcard.__path__[0], 'models/pretrained/uno_dqn')

with sess.as_default():
    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(check_point_path))

# Evaluate the performance. Play with random agents.
evaluate_num = 1000
random_agent = RandomAgent(env.action_num)
env.set_agents([dqn_agents[0], random_agent])
reward = tournament(env, evaluate_num)[0]
print('Average reward against random agent: ', reward)