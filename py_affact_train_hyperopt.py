#!/usr/bin/python3
"""
Main File to start hyperopt the AFFACT Network
"""
import os

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter


from utils.config_utils import get_config, create_result_directory
from training.train_model import TrainModel
from utils.utils import init_environment
from functools import partial
import os
import wandb



def main():
    """
    Run training for a specific model
    Returns
    -------

    """
    # Load configuration for training
    config = get_config('train/affact_hyperopt')
    # Init environment, use GPU if available, set random seed
    device = init_environment(config)
    # Create result directory
    create_result_directory(config)
    # Create a training instance with the loaded configuration on the loaded device
    # training_instance = TrainModel(config, device)
    # Run the training
    # training_instance.train()



    prefix = "/home/yves/Desktop/uzh/affact/PyAffact/"
    print(prefix + config.dataset.dataset_labels_filename)


    def train_hyperopt(config, data_dir=None):
        config.basic.result_directory = prefix + "results"
        config.dataset.dataset_labels_filename = prefix + config.dataset.dataset_labels_filename
        config.dataset.partition_filename = prefix + config.dataset.partition_filename
        config.dataset.dataset_image_folder = prefix + config.dataset.dataset_image_folder
        config.dataset.landmarks_filename = prefix + config.dataset.landmarks_filename
        config.dataset.bounding_boxes_filename = prefix + config.dataset.bounding_boxes_filename
        training_instance = TrainModel(config, device)
        # Run the training
        training_instance.train_resnet_51_hyperopt()


    data_dir = os.path.abspath("./data_dir")
    config.preprocessing.dataloader.batch_size = tune.choice([32, 64])
    wandb.init(config=config)
    # config.training.epochs = tune.choice([1, 2])

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    print("starting")
    result = tune.run(
        partial(train_hyperopt),
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


    # Create the sweep
    wandb.sweep(result)

    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)
    #
    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)
    #
    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))



if __name__ == '__main__':
    main()
