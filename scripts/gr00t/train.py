import tyro
from pathlib import Path
from phosphobot_old.am import Gr00tTrainerConfig, Gr00tTrainer

config = tyro.cli(Gr00tTrainerConfig)

# Get the path to the cloned gr00t repo
path_to_gr00t_folder = Path(__file__).parent.parent.parent / "Isaac-GR00T"
config.training_params.path_to_gr00t_repo = str(path_to_gr00t_folder)

trainer = Gr00tTrainer(config)
trainer.train()

print("Training complete")
