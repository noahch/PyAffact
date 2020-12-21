from evaluation.charts import generate_model_accuracy_of_testsets_2
from preprocessing.dataset_generator import generate_test_dataset
from utils.config_utils import get_config
import pandas as pd

# TODO refactoring: is this file really needed?
def _get_attribute_baseline_accuracy_val(labels, config):
    train_attribute_baseline_majority_value = pd.read_pickle(config.dataset.majority_class_file,
                                                             compression='zip')
    x = labels.apply(pd.Series.value_counts)
    lst = train_attribute_baseline_majority_value.tolist()
    access_tuple_list = [(0 if lst[y] == -1 else 1, y) for y in range(0, len(lst))]
    result_list = []
    for t in access_tuple_list:
        result_list.append((train_attribute_baseline_majority_value.keys()[t[1]], x.iloc[t] / labels.shape[0]))
    return pd.DataFrame(result_list).set_index(0)[1]
