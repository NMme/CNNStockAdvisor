import torch
import pandas as pd
import os

from model import CNN1
from logger import Logger
from data_generator import DataGenerator

pd.options.display.width = 0
company_code = 'GOOGL'
ROOT_PATH = ".."
iter_changes = "fresh_rolling_train"  # label for changes in this run iteration
INPUT_PATH = os.path.join(ROOT_PATH, "stock_history", company_code)
OUTPUT_PATH = os.path.join(ROOT_PATH, "outputs", iter_changes)
LOG_PATH = OUTPUT_PATH + os.sep + "logs"
LOG_FILE_NAME_PREFIX = "log_{}_{}".format(company_code, iter_changes)
PATH_TO_STOCK_HISTORY_DATA = os.path.join(ROOT_PATH, "stock_history")
PATH_TO_COMPANY_DATA = os.path.join(PATH_TO_STOCK_HISTORY_DATA, company_code, company_code + '.csv')

# setup logger and datagenerator
logger = Logger(LOG_PATH, LOG_FILE_NAME_PREFIX)
data_gen = DataGenerator(company_code, PATH_TO_COMPANY_DATA, OUTPUT_PATH, update=False, logger=logger)
# generate data
x_train, y_train, x_cv, y_cv, x_test, y_test, df_batch_train, df_batch_test, sample_weights, is_last_batch = \
    data_gen.get_rolling_data_next(None, 12)

# create model
net = CNN1()
# set optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# for epoch in range(2):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

print('Finished Training')

if __name__ == "__main__":
    # execute only if run as a script
    x_train, y_train, x_cv, y_cv, x_test, y_test, df_batch_train, df_batch_test, sample_weights, is_last_batch = \
        data_gen.get_rolling_data_next(None, 12)
    print(x_train[0,:,:,1])
    print(y_train[0])

