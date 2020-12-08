# PyAffact
PyTorch implementation of the Affact Paper

# CMDs
- conda install -c https://www.idiap.ch/software/bob/conda bob.io.image
- conda install -c https://www.idiap.ch/software/bob/conda bob.ip.base
- conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
- conda install -c plotly plotly-orca

#Today
- baseline

# TODO
- hyperopt, with multiple arguments
- Multi GPU
- continue training
- report
- refactoring
- landmark detector
- color temp shift
- POS_weight with BCEWithLogitsLoss

# Experiments
affact:
    - sgd: 0.1 und denn lr --> 0.01 (ohne momentum)
        --config.name=affact_config_rolf --basic.cuda_device_name=cuda:7 --basic.experiment_name=affactSgdNoMomentumLR001 --training.optimizer.momentum=0
    - adam: 0.1, gamma: 0.1, lr step = 10, 20 epochs
        --config.name=affact_config_rolf --basic.cuda_device_name=cuda:6 --basic.experiment_name=affactAdamLR01 --training.lr_scheduler.step_size=10 --training.optimizer.type=Adam
    - adam: 0.01, 30 epochs
        --config.name=affact_config_rolf --basic.cuda_device_name=cuda:5 --basic.experiment_name=affactAdamLR001 --training.optimizer.type=Adam --training.lr_scheduler.step_size=10 --training.optimizer.learning_rate=0.01
baseline:
    - adam: 0.1
        --config.name=baseline_config_rolf --basic.cuda_device_name=cuda:4 --basic.experiment_name=baselineAdamLR01 --training.optimizer.type=Adam --training.lr_scheduler.step_size=10


# Questions
- evaluation:
    - welche bilder zur evaluation brauchen um performance von unserem und ihres zu vergleichen?
    - erst bei subset netzwerk besser als baseline
    
                        /local/scratch/datasets/CelebA/protocol/list_eval_partition.txt