import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Загрузка модели и скалера
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

# Названия признаков
features = ['ALB', 'BUN', 'A/G', 'GLU', 'GLO', 'BUN|CRE', 'TBIL', 'Na+', 'P', 'AMY']

# Описания признаков
feature_descriptions = {
    'ALB': 'Уровень альбумина в крови',
    'BUN': 'Концентрация уреи в крови',
    'A/G': 'Соотношение альбумина к глобулинам',
    'GLU': 'Уровень глюкозы в крови',
    'GLO': 'Уровень глобулинов в крови',
    'BUN|CRE': 'Соотношение уреи к креатинину',
    'TBIL': 'Уровень общего билирубина',
    'Na+': 'Уровень натрия в крови',
    'P': 'Уровень фосфора в крови',
    'AMY': 'Уровень амилазы в крови'
}

# Заголовок приложения
st.title("Прогнозирование эффекта от введения продукта")
st.subheader("Анализ влияния на биохимические параметры у крупного рогатого скота")

# Описание приложения
st.markdown("""
Это приложение демонстрирует, как введение нашего продукта влияет на биохимические параметры у крупного рогатого скота.
Введите текущие значения параметров животного, чтобы получить прогноз после применения препарата.
""")

# Функция предсказания
def predict_product_effect(input_data):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)
    return prediction, probability

# Интерфейс ввода данных
st.subheader("Введите текущие значения параметров:")

# Создаем форму для ввода данных
user_input = {}
for feature in features:
    with st.expander(f"{feature} - {feature_descriptions[feature]}"):
        user_input[feature] = st.number_input(feature, key=feature)

# Формирование DataFrame из введенных данных
input_df = pd.DataFrame([user_input])

# Кнопка для выполнения предсказания
if st.button('Предсказать эффект'):
    # Выполнение предсказания
    prediction, probability = predict_product_effect(input_df)
    
    # Вывод результатов
    st.subheader("Результаты прогноза:")
    
    if prediction[0] == 1:
        st.success("Прогнозируется положительный эффект от введения продукта!")
        st.write(f"Вероятность эффекта: {probability[0][1]:.2f}")
    else:
        st.warning("Прогнозируется отсутствие эффекта от введения продукта.")
        st.write(f"Вероятность отсутствия эффекта: {probability[0][0]:.2f}")
    
    # Визуализация изменения параметров
    st.subheader("Прогнозируемое изменение параметров:")
    
    # Пример прогнозируемых значений (замените на реальные расчеты)
    predicted_values = {
        'ALB': user_input['ALB'] * 0.95,
        'BUN': user_input['BUN'] * 1.1,
        'A/G': user_input['A/G'] * 0.98,
        'GLU': user_input['GLU'] * 0.97,
        'GLO': user_input['GLO'] * 1.05,
        'BUN|CRE': user_input['BUN|CRE'] * 1.15,
        'TBIL': user_input['TBIL'] * 0.9,
        'Na+': user_input['Na+'] * 0.98,
        'P': user_input['P'] * 1.03,
        'AMY': user_input['AMY'] * 0.85
    }
    
    # Таблица с текущими и прогнозируемыми значениями
    results_df = pd.DataFrame({
        'Параметр': features,
        'Текущее значение': [user_input[feature] for feature in features],
        'Прогнозируемое значение': [predicted_values[feature] for feature in features]
    })
    st.dataframe(results_df)
    
    # Графики изменения параметров
    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(features))
    ax.bar(x, [user_input[feature] for feature in features], width=0.4, label='Текущее значение', align='center')
    ax.bar(x, [predicted_values[feature] for feature in features], width=0.4, label='Прогнозируемое значение', align='edge')
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend()
    ax.set_title('Изменение биохимических параметров после введения продукта')
    st.pyplot(fig)

# Добавляем информацию о параметрах
st.markdown("---")
st.subheader("Информация о параметрах")
for feature in features:
    st.write(f"**{feature}** - {feature_descriptions[feature]}")

# Добавляем информацию о модели
st.markdown("---")
st.subheader("Информация о модели")
st.markdown("""
Модель построена на основе логистической регрессии с взвешиванием классов.
Она обучена на исторических данных с использованием следующих параметров:
- Кросс-валидация: 5-кратная стратифицированная
- Средний AUC-ROC на кросс-валидации: 0.88
- Точность на тестовой выборке: 0.625
- AUC-ROC на тестовой выборке: 0.75
""")

# Добавляем информацию о применении
expander = st.expander("Показать рекомендации по применению продукта")
expander.markdown("""
**Рекомендации по применению продукта:** 

1. **Дозировка**: согласно инструкции производителя (обычно рассчитывается на кг веса животного)
2. **Периодичность**: предпочтительно в периоды повышенной метаболической нагрузки
3. **Противопоказания**: не рекомендуется для животных с критически низкими уровнями альбумина (<30 г/л)
4. **Мониторинг**: рекомендуется контрольное измерение биохимических показателей через 10-14 дней после начала применения

**Экономический эффект**: Применение продукта в рекомендуемых дозах способствует оптимизации обменных процессов, что может привести к улучшению продуктивных показателей животных.
""")

# Добавляем сноску
st.markdown("---")
st.markdown("© 2025 | Информация предназначена только для демонстрационных целей в ветеринарии")
