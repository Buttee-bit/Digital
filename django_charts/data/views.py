import os

import pandas as pd
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect


# Create your views here


def index(request):
    global attributied
    if request.method == "POST":
        uploaded_file = request.FILES['document']
        if uploaded_file.name.endswith('.csv'):
            # save csv file in media folder
            savefile = FileSystemStorage()

            name = savefile.save(uploaded_file.name, uploaded_file)  # this is a name file

            # know where to save file
            d = os.getcwd()  # Current directory of the project
            file_directiry = d + '\media\\' + name
            readfile(file_directiry)
            return redirect(results)
    else:
        messages.warning(request, 'File was not uploaded. Please use csv file extension !')

    return render(request, 'dashboard/index.html')


##project.csv

def readfile(filename):
    ## Объявление глобальных переменных
    global list_keys_valuse_counts, list_values_valuse_counts, \
        data_norm, my_file, list_keys_to_week, \
        list_values_to_week, list_keys_mount, \
        list_values_mount, dict_week

    ## Чтение и обрабока данных
    def normalize_data():
        data = pd.read_csv('data_tg.csv')
        data['date_time'] = data['date'] + ' ' + data['time']
        data['date_time'] = pd.to_datetime(data['date_time'])
        data.index = data['date_time']
        data = data.drop(columns=['date', 'time'], axis=1)
        data = data.drop(columns=['md5'], axis=1)
        data = data.drop(columns=['image'], axis=1)
        data = data.drop(columns=['messageLink'], axis=1)
        data = data.drop(columns=['messageImage'], axis=1)
        data = data.drop(columns=['forwardedChanel'], axis=1)

        return data

    data_norm = normalize_data()

    ##  Создание графика распределения по дням для каждой группы
    def get_list_value_counts(data_norm):
        data_counts = data_norm.groupby('chanel', as_index=False).agg({
            "id": 'count'
        }).sort_values('id', ascending=False)
        list_keys_valuse_counts = []
        list_values_valuse_counts = []
        for key, value in zip(data_counts['chanel'], data_counts['id']):
            list_keys_valuse_counts.append(key)
            list_values_valuse_counts.append(value)
        return list_keys_valuse_counts, list_values_valuse_counts

    list_keys_valuse_counts, list_values_valuse_counts = get_list_value_counts(data_norm)

    ## Создание графика распределения сообщений по дням месяца
    def get_list_key_value_mount(data_norm):
        data_norm_M = data_norm.resample('d').id.nunique().to_frame(name='mount_count')
        data_norm_M['Day'] = data_norm_M.index.day
        list_keys_mount = list(data_norm_M.Day.unique())
        list_values_mount = list(data_norm_M.mount_count)
        return list_keys_mount, list_values_mount

    list_keys_mount, list_values_mount = get_list_key_value_mount(data_norm)

    def value_for_week(data_norm):

        data_to_week = pd.DataFrame(data=data_norm, index=None)
        data_to_week['date_time'] = pd.to_datetime(data_to_week['date_time'])
        data_to_week['dow'] = data_to_week.date_time.dt.day_name()
        data_to_week['hour'] = data_to_week.date_time.dt.hour

        data_to_week = data_to_week.groupby([
            'dow',
            'hour'
        ]).hour.count().to_frame(name='day_hour_count')
        data_to_week = data_to_week.reset_index()
        # Сюда вставить цикл по дням недели
        list_week = list(data_to_week.dow.unique())

        dict_week = {}
        for i in list_week:
            dict_week[i] = [[], []]

        for day in dict_week:
            data_to_day = data_to_week[data_to_week['dow'] == day]
            for key, value in zip(data_to_day['hour'], data_to_day['day_hour_count']):
                dict_week[day][0].append(key)
                dict_week[day][1].append(value)
        return dict_week

    dict_week = value_for_week(data_norm)


def results(request):
    list_k_Monday = dict_week['Monday'][0]
    list_v_Monday = dict_week['Monday'][1]
    list_k_Saturday = dict_week['Saturday'][0]
    list_v_Saturday = dict_week['Saturday'][1]
    list_k_Sunday = dict_week['Sunday'][0]
    list_v_Sunday = dict_week['Sunday'][1]
    list_k_Thursday = dict_week['Thursday'][0]
    list_v_Thursday = dict_week['Thursday'][1]
    list_k_Tuesday = dict_week['Tuesday'][0]
    list_v_Tuesday = dict_week['Tuesday'][1]
    list_k_Wednesday = dict_week['Wednesday'][0]
    list_v_Wednesday = dict_week['Wednesday'][1]
    list_k_Friday = dict_week['Friday'][0]
    list_v_Friday = dict_week['Friday'][1]

    context = {
        'list_keys_mount': list_keys_mount,
        'list_values_mount': list_values_mount,
        'list_keys_valuse_counts': list_keys_valuse_counts,
        'list_values_valuse_counts': list_values_valuse_counts,
        'list_k_Monday': list_k_Monday,
        'list_v_Monday': list_v_Monday,
        'list_k_Saturday': list_k_Saturday,
        'list_v_Saturday': list_v_Saturday,
        'list_k_Sunday': list_k_Sunday,
        'list_v_Sunday': list_v_Sunday,
        'list_k_Thursday': list_k_Thursday,
        'list_v_Thursday': list_v_Thursday,
        'list_k_Tuesday': list_k_Tuesday,
        'list_v_Tuesday': list_v_Tuesday,
        'list_k_Wednesday': list_k_Wednesday,
        'list_v_Wednesday': list_v_Wednesday,
        'list_k_Friday': list_k_Friday,
        'list_v_Friday': list_v_Friday
    }
    return render(request, 'dashboard/results.html', context)

def nlp_results(request):
    list_path_photo = []
    return render(request,'dashboard/nlp_results.html')