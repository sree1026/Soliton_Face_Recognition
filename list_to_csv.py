import csv


def list_2_csv(data):
    """
    This function converts encodings to CSV file.
    :param data: It contains list of names and encodings
    :return: it doesn't return anything
    """
    # sorting the encoding according to alphabetical order.
    data.sort(key=lambda x: x[0])
    names_list = []
    with open('encodings_file_new.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        for encoding in data:
            # converting the 128-d encoding to string
            encoding_list = [str(value) for value in encoding[1]]
            # converting list of string to list of float value
            encoding_list_value = [float(value) for value in encoding_list]
            # converting the name to a list.
            names_list.append(encoding[0])
            row = names_list+encoding_list_value
            writer.writerow(row)
            names_list.pop()

