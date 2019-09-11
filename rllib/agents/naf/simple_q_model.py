from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class SimpleQModel(TFModelV2):
    """Extension of standard TFModel to provide Q values.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_q_values() -> Q(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        head_params={"mu_hiddens": [256], "l_hiddens": [256], "v_hiddens": [256]},
    ):
        """Initialize variables of this model.

        Extra model kwargs:
            q_hiddens (list): defines size of hidden layers for the q head.
                These will be used to postprocess the model output for the
                purposes of computing Q values.

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the Q head. Those layers for forward()
        should be defined in subclasses of SimpleQModel.
        """

        super(SimpleQModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        # setup the Q head output (i.e., model for get_q_values)
        self.model_out = tf.keras.layers.Input(shape=(num_outputs,), name="model_out")
        self.action = tf.keras.layers.Input(shape=(action_space,), name="action")
        head_params["mu_hiddens"].append(action_space)
        self._init_quadratic_networks(**head_params)

        self.q_value = tf.keras.Model(
            [self.model_out, self.action], self.Q
        )
        self.policy = tf.keras.Model(
            self.model_out, self.Q_params['mu']
        )
        self.register_variables(self.q_value.Q)
        
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """This generates the model_out tensor input.

        You must implement this as documented in modelv2.py."""
        raise NotImplementedError

    def get_q_values(self, model_out, action):
        """Returns Q(s, a) given a feature tensor for the state.

        Override this in your custom model to customize the Q output head.

        Arguments:
            model_out (Tensor): embedding from the model layers

        Returns:
            action scores Q(s, a) for each action, shape [None, action_space.n]
        """
        if action is None:
            action = self.policy([model_out])
        return self.q_value([model_out, action])
    
    def get_action(self, model_out):
        return self.policy([model_out])

    def _init_quadratic_networks(self, mu_hiddens, l_hiddens, v_hiddens):
        quadratic_networks = {
            "mu_hidden": mu_hiddens,
            "l_hidden": l_hiddens,
            "v_hidden": v_hiddens,
        }
        self.Q_params = {}
        for key, hiddens in quadratic_networks.items():
            last_layer = self.model_out
            for idx, hidden in enumerate(hiddens):
                last_layer = tf.keras.layers.Dense(
                    hidden, name="{}_{}".format(key, idx), activation=None
                )(last_layer)
                if idx != len(hiddens) - 1:
                    last_layer = tf.keras.activations.relu(last_layer)
            self.Q_params[key.split("_")[0]] = last_layer
        self.Q_params["mu"] = tf.keras.activations.tanh(self.Q_params["mu"])
        diagonal = tf.linalg.diag(tf.exp(tf.linalg.diag_part(self.Q_params["l"])))
        L = diagonal + tf.linalg.band_part(self.Q_params["l"], 0, -1)
        P = tf.matmul(L, tf.transpose(L))
        diff = tf.keras.layers.subtract([action, self.Q_params["mu"]])
        self.Q = tf.math.scalar_mul(-0.5, tf.matmul(tf.matmul(tf.transpose(diff), P), diff))
