import bob
import pandas as pd

# Load partition file
from torchvision.transforms import transforms

from utils.config_utils import get_config
from evaluation.utils import tensor_to_image
from preprocessing.affact_transformer import AffactTransformer

config = get_config('testsetA_config')

partition_file_path = 'dataset/CelebA/list_eval_partition.txt'
labels_file_path = 'dataset/CelebA/list_attr_celeba.txt'
labels = pd.read_csv(labels_file_path, delim_whitespace=True, skiprows=1)

partition_df = pd.read_csv(partition_file_path, delim_whitespace=True, header=None)
partition_df.columns = ['filename', 'partition']
partition_df["partition"] = pd.to_numeric(partition_df["partition"])

test_partition_df = partition_df.loc[partition_df["partition"] == 2]

df_test_labels = pd.merge(labels, test_partition_df, left_index=True, right_on='filename', how='inner')
df_test_labels.set_index('filename', inplace=True)
df_test_labels.drop(columns=["partition"], inplace=True)
landmarks = pd.read_csv(config.preprocessing.dataset.landmarks_filename,
                        delim_whitespace=True, skiprows=1)

df_test_landmarks = pd.merge(landmarks, test_partition_df, left_index=True, right_on='filename', how='inner')
df_test_landmarks.set_index('filename', inplace=True)



# define transformer for each set


# align with handlabeld landmarks

data_transforms_A = transforms.Compose([AffactTransformer(config)])


image = bob.io.base.load('{}/{}'.format(config.preprocessing.dataset.dataset_image_folder, '202592.jpg'))
landmarks, bounding_boxes = None, None
if config.preprocessing.dataset.uses_landmarks:
    landmarks = df_test_landmarks.loc['202592.jpg'].tolist()
    landmarks = landmarks[:4] + landmarks[6:]

input = {
    'image': image,
    'landmarks': landmarks,
    'bounding_boxes': bounding_boxes,
    'index': 0
}
# print(self.labels.iloc[index].name)
X, bbx = data_transforms_A(input)

img = tensor_to_image(X)
img.show()
# face detector -> grösser bbx -> 10crop

# face detector -> bounding box -> transformation






# read data and generate folder with transformed images

