import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats 


@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('cp1251')


def test_hypothesis(sample_1, sample_2, alpha):
    n_1, n_2 = len(sample_1), len(sample_2)
    if n_1 < 2 or n_2 < 2:
        st.write('В выборке меньше 2 человек, удовлетворящих заданному возрасту и/или количеству пропущенных дней. Измените эти параметры')
        return
    mean_1, mean_2 = np.mean(sample_1), np.mean(sample_2)
    ddof_1 = 1 if n_1 > 1 else 0
    ddof_2 = 1 if n_2 > 1 else 0
    var_1, var_2 = np.var(sample_1, ddof=ddof_1), np.var(sample_2, ddof=ddof_2)

    # pooled sample variance
    var = ( ((n_1-1)*var_1) + ((n_2-1)*var_2 ) ) / (n_1+n_2-2)
    std_error = np.sqrt(var * (1.0 / n_1 + 1.0 / n_2))
    #st.write('mean_1, mean_2, var_1, var_2 , var, std_error = ', mean_1, mean_2, var_1, var_2 , var, std_error)
    # calculate t statistics
    t = abs(mean_1 - mean_2) / std_error
    st.write('t-статистика:',t)

    dof = n_2+n_1-2
    
    t_c = stats.t.ppf(q=1-alpha, df=dof)
    st.write("Критическое значение для односторонней гипотезы:",t_c)

    p_one = 1-stats.t.cdf(x=t, df=dof)
    st.write("P-значение:", p_one)
    if p_one > alpha:
        st.write(f'{t_c} > {t}')
        st.write(f'({p_one} > {alpha})')
        st.write('Нет оснований отвергать гипотезу')
    else:
        st.write(f'{t_c} < {t}')
        st.write(f'({p_one} < {alpha})')
        st.write('Гипотеза отвергается в пользу альтернативы')


df = pd.read_excel('Raw\\m_tech\\stats.xlsx',header=0)
csv = convert_df(df)
st.download_button("Press to Download", csv, "file.csv", "text/csv", key='download-csv')

df[['Количество больничных дней', 'Возраст', 'Пол']] = pd.DataFrame(df['Количество больничных дней,Возраст,Пол'].str.split(',').tolist())
df = df[['Количество больничных дней', 'Возраст', 'Пол']]
dict_encoder = {'"М"':0, '"Ж"':1}
df['Пол'] = df['Пол'].map(dict_encoder)
df['Количество больничных дней'] = df['Количество больничных дней'].astype(int)
df['Возраст'] = df['Возраст'].astype(int)

work_days = st.number_input('Выберите количество пропущенных по болезни рабочих дней', min_value=0, max_value=365)

st.write(f'Гипотеза №1: Нет статистически значимой разницы в пропусках более {work_days} дней по болезни между мужчинами и женщинами')
st.write(f'Альтернатива: Мужчины пропускают в течение года более {work_days} рабочих дней по болезни значимо чаще женщин')


st.write('Распределение количества больничных дней среди женщин')
fem_workdays = df[df['Пол']==1]['Количество больничных дней']
fem_hist = fem_workdays.value_counts().sort_index()
st.bar_chart(data = fem_hist)

st.write('Распределение количества больничных дней среди женщин')
male_workdays = df[df['Пол']==0]['Количество больничных дней']
male_hist = male_workdays.value_counts().sort_index()
st.bar_chart(data = male_hist)



male_sample = df[(df['Количество больничных дней']>work_days) & (df['Пол']==0)]['Количество больничных дней']
female_sample = df[(df['Количество больничных дней']>work_days) & (df['Пол']==1)]['Количество больничных дней']
alpha = st.number_input('Выберите уровень значимости', value=0.05, min_value = 0., max_value = 1., key=0)
test_hypothesis(male_sample, female_sample, alpha)


age = st.number_input('Выберите возраст', min_value = df['Возраст'].min(), max_value = df['Возраст'].max()-1)
st.write(f'Гипотеза №2: Нет статистически значимой разницы в пропусках {work_days} дней по болезни между людьми старше и младше {age} лет')
st.write(f'Альтернатива: Работники старше {age} лет пропускают в течение года более {work_days} рабочих дней по болезни значимо чаще своих более молодых коллег.')


st.write(f'Распределение количества больничных дней среди людей младше или {age} лет')
young_workdays = df[df['Возраст']<=age]['Количество больничных дней']
young_hist = young_workdays.value_counts().sort_index()
st.bar_chart(data = young_hist)

st.write(f'Распределение количества больничных дней среди людей старше {age} лет')
old_workdays = df[df['Возраст']>age]['Количество больничных дней']
old_hist = old_workdays.value_counts().sort_index()
st.bar_chart(data = old_hist)


young_sample = df[(df['Количество больничных дней']>work_days) & (df['Возраст']<=age)]['Количество больничных дней']
old_sample = df[(df['Количество больничных дней']>work_days) & (df['Возраст']>age)]['Количество больничных дней']
alpha = st.number_input('Выберите уровень значимости', value=0.05, min_value = 0., max_value = 1., key=1)
test_hypothesis(young_sample, old_sample, alpha)
