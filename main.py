#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import hydra
from hydra.core.config_store import ConfigStore
from src.config import MyConfig
from src.experiments import Experiment
import torch


cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


@hydra.main(config_path="src/conf", config_name="config_vat", version_base="1.1")
def main(cfg: MyConfig) -> None:
    experiment = Experiment(cfg)
    experiment.setup()
    torch.autograd.set_detect_anomaly(True)
    experiment.run()


if __name__ == "__main__":
    main()
