BATCH_SIZE = 16
EPOCH_NUMBER = 150
DATASET = ['G-U', 2]
# DATASET = ['Class4_labeled', 2]

crop_size = (1024, 1024)

class_dict_path = './Datasets/' + DATASET[0] + '/class_dict.csv'
TRAIN_ROOT = './Datasets/' + DATASET[0] + '/train/image'
TRAIN_LABEL = './Datasets/' + DATASET[0] + '/train/label'

VAL_ROOT = './Datasets/' + DATASET[0] + '/val/image'
VAL_LABEL = './Datasets/' + DATASET[0] + '/val/image'

TEST_ROOT = './Datasets/' + DATASET[0] + '/predict/image'
TEST_LABEL = './Datasets/' + DATASET[0] + '/predict/label'



