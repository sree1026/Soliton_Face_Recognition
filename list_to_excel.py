import pickle
import os
import numpy as np
import pandas as pd
import xlsxwriter as xls

data = pickle.loads(open('encodings_dlib', 'rb').read())
train_image_encodings = data
# encoding = []
wb = xls.Workbook('output2.xlsx')
ws = wb.add_worksheet("New Sheet")
for row, item in enumerate(train_image_encodings):
    name = ''.join(filter(lambda x: not x.isdigit(), item[0]))
    ws.write_string(row, 0, name)
    # print(row)
    for column, encoding in enumerate(item[1]):
        ws.write(row, column+1, encoding)
        # print(str(column) + " :: " + str(encoding))

wb.close()
# data = pd.DataFrame(train_image_encodings).to_excel('output.xlsx', header=False, index=False)
