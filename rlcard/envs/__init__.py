''' Register new environments
'''
from rlcard.envs.env import Env
from rlcard.envs.registration import register, make

register(
    env_id='wizard_trickpreds',
    entry_point='rlcard.envs.wizard_trickpreds:WizardEnv',
)

register(
    env_id='wizard',
    entry_point='rlcard.envs.wizard:WizardEnv',
)

register(
    env_id='wizard_simple',
    entry_point='rlcard.envs.wizard_simple:WizardEnv',
)

register(
    env_id='wizard_most_simple',
    entry_point='rlcard.envs.wizard_most_simple:WizardEnv',
)

register(
    env_id='wizard_ms_trickpreds',
    entry_point='rlcard.envs.wizard_ms_trickpreds:WizardEnv',
)

register(
    env_id='wizard_s_trickpreds',
    entry_point='rlcard.envs.wizard_s_trickpreds:WizardEnv',
)

register(
    env_id='wizard_ms_trickpreds_with_humans',
    entry_point='rlcard.envs.wizard_ms_trickpreds_with_humans:WizardEnv',
)

register(
    env_id='wizard_s_trickpreds_with_humans',
    entry_point='rlcard.envs.wizard_s_trickpreds_with_humans:WizardEnv',
)
