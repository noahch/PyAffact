from config.config_utils import get_config
from training.train_model import TrainModel
from utils.utils import init_environment

# Init environment, use GPU if available, set random seed
device = init_environment()

# Load configuration for training
config = get_config("basic_config")

# Create a training instance with the loaded configuration on the loaded device
training_instance = TrainModel(config, device)

# Run the training
training_instance.train()

