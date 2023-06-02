# import os
import pickle

class Agent:
    def __init__(self):
        # If you need to load your model from a file this is the time to do it
        # You can do something like:
        #
        # self.actor = # your Model
        #
        # cur_dir = os.path.dirname(os.path.realpath(__file__))
        # with open(os.path.join(cur_dir, 'model.p'), 'rb') as file:
        #     model = pickle.load(file)
        # self.actor.load_state_dict(model)
        with open("WLTR/src/model.p", "rb") as file:
            model = pickle.load(file)
        self.actor = model
        pass

    def act(self, state):
        # Evaluate your model here
        action, _ = self.actor.predict(state)
        return action
