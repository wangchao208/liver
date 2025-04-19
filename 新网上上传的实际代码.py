import pandas as pd
from sklearn.naive_bayes import GaussianNB
import streamlit as st
import joblib

# 读取训练集数据
train_data = pd.read_csv('训练集.cvs')

# 分离输入特征和目标变量
X = train_data[['Age', 'T',
                'N', 'bone', 'Distant.LN..2016..', 'surgery', 'Radiation.recode', 'brain']]
y = train_data['lung']

# 创建并训练GBM模型
gbm_model = GaussianNB(var_smoothing=1e-9)
gbm_model.fit(X, y)


# 特征映射
feature_order = [
    'Age', 'T', 'N', 'bone',
    'Distant.LN..2016..', 'surgery', 'Radiation.recode', 'brain'
]
class_mapping = {0: "No Lung metastasis", 1: "Live cancer lung metastasis"}
Age_mapper = {"18-66 years": 1, "67-74 years": 2, "＞74 years": 3}
bone_mapper = {"No": 0, "Yes": 1}
Distant_LN_2016__mapper = {"No": 0, "Yes": 1}
T_mapper = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
N_mapper = {"N0": 0, "N1": 1}
Surgery_mapper = {"NO": 0, "Yes": 1}
Radiation_mapper = {"NO": 0, "Yes": 1}
brain_mapper = {"NO": 0, "Yes": 1}


# 预测函数
def predict_lung_metastasis(Age, T,
                            N, bone, Distant_LN_2016_, surgery, Radiation_recode, brain):
    input_data = pd.DataFrame({
        'Age': [Age_mapper[Age]],
        'T': [T_mapper[T]],
        'N': [N_mapper[N]],
        'bone': [bone_mapper[bone]],
        'Distant.LN..2016..': [Distant_LN_2016__mapper[Distant_LN_2016_]],
        'surgery': [Surgery_mapper[surgery]],
        'Radiation.recode': [Radiation_mapper[Radiation_recode]],
        'brain': [brain_mapper[brain]]
    }, columns=feature_order)

    prediction = gbm_model.predict(input_data)[0]
    probability = gbm_model.predict_proba(input_data)[0][1]  # 获取属于类别1的概率
    class_label = class_mapping[prediction]
    return class_label, probability

# 创建Web应用程序
st.title("GBM Model Predicting Lung Metastasis of Cancer")
st.sidebar.write("Variables")

Age = st.sidebar.selectbox("Age", options=list(Age_mapper.keys()))
T = st.sidebar.selectbox("T", options=list(T_mapper.keys()))
N = st.sidebar.selectbox("N", options=list(N_mapper.keys()))
bone = st.sidebar.selectbox("bone", options=list(bone_mapper.keys()))
Distant_LN_2016_ = st.sidebar.selectbox("Distant.LN..2016..", options=list(Distant_LN_2016__mapper.keys()))
surgery = st.sidebar.selectbox("surgery", options=list(Surgery_mapper.keys()))
Radiation_recode = st.sidebar.selectbox("Radiation.recode", options=list(Radiation_mapper.keys()))
brain = st.sidebar.selectbox("brain", options=list(brain_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_lung_metastasis(Age, T,
                                                      N, bone, Distant_LN_2016_, surgery, Radiation_recode, brain)

    st.write("Class Label: ", prediction)  # 结果显示在右侧的列中
    st.write("Probability of developing lung metastasis: ", probability)  # 结果显示在右侧的列中
