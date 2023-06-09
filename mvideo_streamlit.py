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

    var = ( ((n_1-1)*var_1) + ((n_2-1)*var_2 ) ) / (n_1+n_2-2)
    std_error = np.sqrt(var * (1.0 / n_1 + 1.0 / n_2))
    t = abs(mean_1 - mean_2) / std_error
    st.write('t-статистика:',t)

    dof = n_2+n_1-2
    
    t_c = stats.t.ppf(q=1-alpha, df=dof)
    st.write("Критическое значение для односторонней гипотезы:",t_c)

    p = 1-stats.t.cdf(x=t, df=dof)
    st.write("P-значение:", p)
    if p > alpha:
        st.write(f'{t_c} > {t}')
        st.write(f'({p} > {alpha})')
        st.markdown('Вывод: **нет оснований отвергать гипотезу**')
    else:
        st.write(f'{t_c} < {t}')
        st.write(f'({p} < {alpha})')
        st.markdown('Вывод: **гипотеза отвергается в пользу альтернативы**')


df = pd.read_excel('stats.xlsx',header=0)
csv = convert_df(df)
st.download_button("Press to Download", csv, "file.csv", "text/csv", key='download-csv')

df[['work_days', 'age', 'sex']] = pd.DataFrame(df['Количество больничных дней,Возраст,Пол'].str.split(',').tolist())
df = df[['work_days', 'age', 'sex']]

dict_encoder = {'"М"':0, '"Ж"':1}
df['sex'] = df['sex'].map(dict_encoder)
df['work_days'] = df['work_days'].astype(int)
df['age'] = df['age'].astype(int)

work_days = st.number_input('Выберите количество пропущенных по болезни рабочих дней', min_value=0, max_value=365)

st.write(f'Гипотеза №1: Нет статистически значимой разницы в пропусках более {work_days} дней по болезни между мужчинами и женщинами')
st.write(f'Альтернатива: Мужчины пропускают в течение года более {work_days} рабочих дней по болезни значимо чаще женщин')


st.write('Распределение количества больничных дней среди женщин')
fem_workdays = df[df['sex']==1]['work_days']
fem_hist = fem_workdays.value_counts().sort_index()
st.bar_chart(data = fem_hist)

st.write('Распределение количества больничных дней среди мужчин')
male_workdays = df[df['sex']==0]['work_days']
male_hist = male_workdays.value_counts().sort_index()
st.bar_chart(data = male_hist)


male_sample = df[(df['work_days']>work_days) & (df['sex']==0)]['work_days']
female_sample = df[(df['work_days']>work_days) & (df['sex']==1)]['work_days']
alpha = st.number_input('Выберите уровень значимости', value=0.05, min_value = 0., max_value = 1., key=0)
test_hypothesis(male_sample, female_sample, alpha)


age = st.number_input('Выберите возраст', min_value = df['age'].min(), max_value = df['age'].max()-1)
st.write(f'Гипотеза №2: Нет статистически значимой разницы в пропусках более {work_days} дней по болезни между людьми старше и младше {age} лет')
st.write(f'Альтернатива: Работники старше {age} лет пропускают в течение года более {work_days} рабочих дней по болезни значимо чаще своих более молодых коллег.')


st.write(f'Распределение количества больничных дней среди людей младше или {age} лет')
young_workdays = df[df['age']<=age]['work_days']
young_hist = young_workdays.value_counts().sort_index()
st.bar_chart(data = young_hist)

st.write(f'Распределение количества больничных дней среди людей старше {age} лет')
old_workdays = df[df['age']>age]['work_days']
old_hist = old_workdays.value_counts().sort_index()
st.bar_chart(data = old_hist)


young_sample = df[(df['work_days']>work_days) & (df['age']<=age)]['work_days']
old_sample = df[(df['work_days']>work_days) & (df['age']>age)]['work_days']
alpha = st.number_input('Выберите уровень значимости', value=0.05, min_value = 0., max_value = 1., key=1)
test_hypothesis(young_sample, old_sample, alpha)
