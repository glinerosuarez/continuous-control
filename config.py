from tools.nets import Activation
from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml'],
    validators=[Validator("ActorCritic.activation", is_in=Activation.get_values())]
)

settings.validators.validate()

assert isinstance(settings.env_file, str)
assert isinstance(settings.train_mode, bool)
assert isinstance(settings.seed, int)
assert isinstance(settings.cores, int)
assert isinstance(settings.out_dir, str)
assert isinstance(settings.ActorCritic.layers, int)
assert isinstance(settings.ActorCritic.activation, str)
assert isinstance(settings.ActorCritic.hidden_nodes, int)
