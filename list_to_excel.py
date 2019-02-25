import xlsxwriter as xls
import xlrd
import csv


def list_2_excel(data):
    data.sort(key=lambda x: x[0])
    # encoding = []
    filename = 'output1.xlsx'
    wb = xls.Workbook(filename)
    ws = wb.add_worksheet("New Sheet")
    for row, item in enumerate(data):
        name = ''.join(filter(lambda x: not x.isdigit(), item[0]))
        ws.write_string(row, 0, name)
        # print(row)
        for column, encoding in enumerate(item[1]):
            ws.write(row, column+1, encoding)
            # print(str(column) + " :: " + str(encoding))

    wb.close()
    return filename


def csv_from_excel(filename):
    wb = xlrd.open_workbook(filename)
    sh = wb.sheet_by_name('New Sheet')
    your_csv_file = open('output4.csv', 'w')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()