import os

import rlcard
from rlcard.agents import CFRAgent
from rlcard.agents import DQNAgent
from rlcard.models.model import Model

# Root path of pretrianed models
ROOT_PATH = os.path.join(rlcard.__path__[0], 'models\pretrained')

class UnoDQNModel(Model):
    ''' A pretrained model on Uno with DQN
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        import tensorflow as tf
        from rlcard.agents import DQNAgent
        #tf.compat.v1.global_variables_initializer()
        #tf.compat.v1.local_variables_initializer()
        
        env = rlcard.make('uno')
        
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        

        with self.graph.as_default():

            self.dqn_agents = []
            for i in range(env.player_num):
                agent = DQNAgent(self.sess,
                    scope='dqn' + str(i),
                    action_num=env.action_num,
                    state_shape=env.state_shape,
                    mlp_layers=[512,512])
                self.dqn_agents.append(agent)
        
       
        check_point_path = os.path.join(ROOT_PATH, 'uno_dqn')
        with self.sess.as_default():
            with self.graph.as_default():
                saver = tf.train.Saver()
#                saver = tf.train.Saver()
                saver.restore(self.sess, tf.train.latest_checkpoint(check_point_path))
        
        
        
    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.dqn_agents