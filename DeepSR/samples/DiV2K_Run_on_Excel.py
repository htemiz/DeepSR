import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mtick
import numpy as np
from os.path import dirname, basename, join, abspath, exists
import csv


dpi = 300


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['figure.dpi'] = dpi
plt.rcParams['savefig.dpi'] = dpi
plt.rcParams['grid.color'] = 'tab:gray'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.1
plt.rcParams['figure.figsize'] = [8.0, 5.0]
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 'x-small'
plt.rcParams['figure.titlesize'] = 'medium'
plt.rcParams['xtick.labelsize'] = "small"
plt.rcParams['ytick.labelsize'] = "small"
# plt.grid(True)


n_methods = 7
sheet_name = 'Mean'
index_columns = [ 'Model Name', 'Learning Rate', 'Normalization', 'Batch Size', 'Input Size', 'Stride', 'Measure']


model_names = ['EDSR_divide 255.0_0.001_24_13_16',
               'EDSR_divide 255.0_0.001_24_23_16',
               'EDSR_divide 255.0_0.001_48_13_16',
               'EDSR_divide 255.0_0.001_48_23_16',
               'EDSR_divide 255.0_0.0001_24_13_16',
               'EDSR_divide 255.0_0.0001_24_23_16',
               'EDSR_divide 255.0_0.0001_48_13_16',
               'EDSR_divide 255.0_0.0001_48_23_16',
               'EDSR_minmax -1 1_0.001_24_13_16',
               'EDSR_minmax -1 1_0.001_24_23_16',
               'EDSR_minmax -1 1_0.001_48_13_16',
               'EDSR_minmax -1 1_0.001_48_23_16',
               'EDSR_minmax -1 1_0.0001_24_13_16',
               'EDSR_minmax -1 1_0.0001_24_23_16',
               'EDSR_minmax -1 1_0.0001_48_13_16',
               'EDSR_minmax -1 1_0.0001_48_23_16',
               'EDSR_divide 255.0_0.001_24_13_32',
               'EDSR_divide 255.0_0.001_24_23_32',
               'EDSR_divide 255.0_0.001_48_13_32',
               'EDSR_divide 255.0_0.001_48_23_32',
               'EDSR_divide 255.0_0.0001_24_13_32',
               'EDSR_divide 255.0_0.0001_24_23_32',
               'EDSR_divide 255.0_0.0001_48_13_32',
               'EDSR_divide 255.0_0.0001_48_23_32',
               'EDSR_minmax -1 1_0.001_24_13_32',
               'EDSR_minmax -1 1_0.001_24_23_32',
               'EDSR_minmax -1 1_0.001_48_13_32',
               'EDSR_minmax -1 1_0.001_48_23_32',
               'EDSR_minmax -1 1_0.0001_24_13_32',
               'EDSR_minmax -1 1_0.0001_24_23_32',
               'EDSR_minmax -1 1_0.0001_48_13_32',
               'EDSR_minmax -1 1_0.0001_48_23_32'
               ]

col_names = [ 'weights.' + str(str(x)).zfill(2) for x in range(1,21,1)]

metrics = ['ERGAS', 'PAMSE', 'PSNR', 'SAM', 'SCC', 'SSIM', 'UQI', 'VIF']  # image quality measures (IQM). DeepSR can measure 18 IQMs


working_dir = r'div2k'
sub_path = r'output\2'
postfix = '_sub_test_results_scale_2.xlsx'




def get_max_index(serie, row_name='PSNR'):
    """

    :param xl:
    :param row_name:
    :return: name of column (str), and column index as number (str)
    """

    # file = join(working_folder, file_pre_name + region + "_" + region + file_rest_name)

    name = serie.idxmax(axis=1).iloc[2]  # PSNR

    idx = name.split('.')[1]

    return name, idx

def get_best_epochs(df, n_measures):

    n_rows = df.shape[0]

    columns= list()

    for i in range(0,n_rows, n_measures):

        serie = df.iloc[i: i+ n_measures, :]

        column, _  = get_max_index(serie)

        columns.append(column)


    return columns



def unique_index_names(xl):
    """
    Returns the index names without NaN values inside
    :param xl: Pandas DataFrame
    :return: a list of index labels
    """
    ivalues = xl.index.values.tolist()

    indexes = list()
    for i in ivalues:
        if i[0] is not np.nan:
            v = list(i)
            # v.pop()  # drop the last index (column) since it is not necessary
            indexes.append(v)

    return indexes


def drop_last_in_list(lst):
    """
    Returns the index names without NaN values inside
    :param xl: Pandas DataFrame
    :return: a list of index labels
    """
    new_list = list()

    for i in lst:
        i.pop()
        new_list.append(i)

    return new_list


def set_index_labels(xl, names):
    """

    :param xl:
    :param names:
    :return:
    """
    array = np.array(names)
    inames = xl.index.names

    mi = pd.MultiIndex.from_arrays(array.transpose(), names=inames)
    # i = pd.Index(names)
    xl.index = mi


def full_path(model_name, test_name=None, postfix=None):

    file = join(abspath(working_dir), model_name)
    file = join(file, sub_path)

    if test_name is not None:

        if postfix is not None:
            file_name = model_name + test_name + postfix
        else:
            file_name = model_name + test_name + ".xlsx"

    else:

        if postfix is not None:
            file_name = model_name + postfix
        else:
            file_name = model_name + ".xlsx"

    file = join(file, file_name)
    return file


def get_ExcelTable(file, region):

    xl = pd.read_excel(file, usecols="A:U", sheet_name=sheet_name, nrows=9)
    xl.index.set_names('Measure', inplace=True)

    # xl.rename({'Unnamed: 0': 'Metric'}, axis='columns', inplace=True)  # rename first column to 'Metric'

    norm, lrate, inp_size, stride, batch_size = parse_model_name(region)

    xl.insert(0, 'Batch Size', batch_size)
    xl.insert(0, 'Stride', stride)
    xl.insert(0, 'Input Size', inp_size)
    xl.insert(0, 'Learning Rate', lrate)
    xl.insert(0, 'Normalization', norm)
    xl.insert(0, 'Model Name', region)

    xl.reset_index(inplace=True)
    xl.set_index(index_columns, inplace=True)

    return xl


def parse_model_name(model_name):
    parts = model_name.split('_')
    # parts[0] =model_name  # return full name
    return parts[1:]  # not necessary return model' name. other parts only

def collect_tables():
    """
    Gathers the 'Mean' Results of models into an Excel File

    :return:
    """
    df = pd.DataFrame(columns=index_columns + col_names)
    df.set_index(index_columns, inplace=True)

    for model_name in model_names:

        norm, lrate, inp_size, stride, batch_size = parse_model_name(model_name)
        file_path = full_path(model_name, postfix)

        if not exists(file_path):
            continue


        xl = get_ExcelTable(file_path, model_name)

        df = df.append(xl)



    df.to_excel('tumu.xlsx')

def gather_best_epochs(df, columns):
    ixs = index_columns.copy()
    # ixs.pop()  # drop 'Measure'

    n_measures = len(metrics)

    # new_df= pd.DataFrame().reindex_like(df)
    new_df = df[0:0].copy()

    new_df.drop(new_df.columns.values, axis=1, inplace=True)

    for i in range(0,len(columns)):
        base = i * n_measures
        ixs = index_columns.copy()
        ixs.append(columns[i])

        tmp = df.loc[base : base + n_measures -1, ixs]
        tmp.rename(columns={columns[i]: 'Scores'}, inplace=True)
        # ixs.pop()

        new_df = new_df.append(tmp)

    return  new_df

def fill_indexes(df):

    df.reset_index(inplace=True)
    ixs = index_columns.copy()
    ixs.pop()  # drop 'Measure'

    n_measures = len(metrics)

    for i in range(0, df.shape[0]):

        base = divmod(i , n_measures)[0] *n_measures

        if i > 0 and  (i % n_measures) !=0 :
            df.loc[i, ixs] = df.loc[base, ixs]

    # df.set_index(ixs, inplace=True)



def plot_graphs():
    file = r"D:\calismalarim\Program\DeepSR\DeepSR\samples\tumu.xlsx"

    xl = pd.read_excel(file, usecols="A:AA", sheet_name='Sheet1')
    xl.set_index(index_columns, inplace=True)

    best_cols = get_best_epochs(xl, len(metrics))

    fill_indexes(xl)
    bests = gather_best_epochs(xl, best_cols)


    bests.to_excel('Best Scores.xlsx')

    bst = bests.copy()
    bst.reset_index()
    bst.set_index(index_columns, inplace=True)
    bst.unstack(level=-1).to_excel('Best Scores, unstacked.xlsx')

    psnrs = bests[bests['Measure'] == 'PSNR']

    names = unique_index_names(xl)
    names = drop_last_in_list(names)

    set_index_labels(psnrs, names)



    return xl

def set5_and_set14():
    """
       Gathers the 'Mean' Results of models into an Excel File

       :return:
       """


    working_dir = r'div2k\_test_bitti'
    sub_path = r'output\2'
    postfix = '_results_scale_2.xlsx'


    for test in ['set5', 'set14']:

        excel_file = test + ', All Results.xlsx'



        df = pd.DataFrame(columns=index_columns + col_names)
        df.set_index(index_columns, inplace=True)

        for model_name in model_names:

            norm, lrate, inp_size, stride, batch_size = parse_model_name(model_name)
            file_path = full_path(model_name, '_' + test, postfix)

            if not exists(file_path):
                continue

            xl = get_ExcelTable(file_path, model_name)

            df = df.append(xl)

        #df.to_excel(excel_file)


        # take best values from the file


        xl = pd.read_excel(excel_file, usecols="A:AA", sheet_name='Sheet1')
        xl.set_index(index_columns, inplace=True)

        best_cols = get_best_epochs(xl, len(metrics))

        fill_indexes(xl)
        bests = gather_best_epochs(xl, best_cols)

        bests.to_excel(test +', Best Scores.xlsx')

        bst = bests.copy()
        bst.reset_index()
        bst.set_index(index_columns, inplace=True)
        bst.unstack(level=-1).to_excel(test + ', Best Scores, unstacked.xlsx')


# collect_tables()

# plot_graphs()



set5_and_set14()






#
# if __name__ == "main":
#     # plot_DL_only()
#
#     collect_tables()
#
#









