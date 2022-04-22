# File path for csv file containing review and it's sentiment as positive, negative or neutral
TRAIN_FILE_PATH = "./Assets/imdb_reviews.csv"

# File path of csv file to test the model 
TEST_FILE_PATH = "./Assets/test_reviews.csv"

# word indexing for words
WORD_INDEX_PATH = "./Assets/word_indexes.csv"

# Device to be choosen as GPU or CPU
DEVICE = "GPU"

# Probability above this value will be considered as positive review
POSITIVE_CAP = 0.53

# Probability below this value will be considered as negative review and above it but below positive cap
# will be considered neutral review
NEUTRAL_CAP = 0.4


