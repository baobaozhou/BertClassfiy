from similarity import BertSim
import tensorflow as tf

bs = BertSim()
bs.set_mode(tf.estimator.ModeKeys.TRAIN)
bs.train()