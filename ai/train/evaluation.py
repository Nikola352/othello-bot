import random

from environment.environment import EnvState


def random_strategy(state: EnvState):
    actions = state.get_available_actions()
    return random.choice(actions) if actions else None

