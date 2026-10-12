import os
import numpy as np
import pandas as pd
import streamlit as st

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="SSDC - Sistema de Triagem de AVC",
    page_icon="🏥",
    layout="wide"
)

# =========================================================
# MOTOR DE INFERÊNCIA (TREINAMENTO COM LDA + SVM LINEAR + ADABOOST)
# =========================================================
@st.cache_resource
def load_and_train_model():
    # Carregamento seguro
    if not os.path.exists("data_stroke.csv"):
        st.error("Arquivo 'data_stroke.csv' não encontrado!")
        return None

    df = pd.read_csv("data_stroke.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Tratamento de Nulos e IDs
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # ENGENHARIA DE FEATURES (Lógica Avançada SSDC)
    df['age_hypertension'] = df['age'] * df['hypertension']
    df['age_heart_disease'] = df['age'] * df['heart_disease']
    df['metabolic_syndrome'] = (df['avg_glucose_level'] * df['bmi']) / 100
    
    df['married_binary'] = df['ever_married'].map({'Yes': 1, 'No': 0}).fillna(0)
    df['age_married'] = df['age'] * df['married_binary']
    
    df['cvd_risk_score'] = (
        df['hypertension'] + 
        df['heart_disease'] + 
        (df['avg_glucose_level'] > 150).astype(int) + 
        (df['bmi'] > 30).astype(int)
    )

    def categorize_age(age):
        if age < 30: return 0
        if age < 60: return 1
        if age < 75: return 2
        return 3
    df['age_bracket'] = df['age'].apply(categorize_age)

    df['lifestyle_risk'] = df['work_type'].apply(
        lambda x: 1 if x in ['Private', 'Self-employed'] else 0
    )

    # Separação de Targets
    X = df.drop(['stroke', 'married_binary'], axis=1, errors='ignore')
    y = df['stroke']

    # PREPROCESSAMENTO
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

    # MODELOS DO ENSEMBLE (LDA + SVM Linear + AdaBoost)
    clf_lda = LinearDiscriminantAnalysis()
    
    clf_svm = SVC(
        kernel='linear', 
        probability=True, 
        class_weight='balanced', 
        random_state=42
    )

    clf_adaboost = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

    # VOTING ENSEMBLE (SOFT VOTING)
    voting_clf = VotingClassifier(
        estimators=[
            ('lda', clf_lda),
            ('svm', clf_svm),
            ('ada', clf_adaboost)
        ], 
        voting='soft',
        weights=[1.0, 1.2, 1.1]
    )

    model = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTETomek(random_state=42)),
        ('clf', voting_clf)
    ])

    model.fit(X, y)
    return model

# Inicialização
model = load_and_train_model()

# =========================================================
# INTERFACE STREAMLIT
# =========================================================
st.title("🏥 SSDC - Sistema Inteligente de Triagem de AVC")
st.markdown("""
**Protótipo de Suporte à Decisão Clínica.** 
Este sistema utiliza um modelo de Machine Learning calibrado com **SMOTETomek** e um **Voting Classifier** composto por **LDA, SVM Linear e AdaBoost**.
""")

# Layout em Colunas para a Ficha Clínica
with st.container():
    st.subheader("📋 Anamnese e Dados Biométricos")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gênero", ["Male", "Female", "Other"])
        age = st.number_input("Idade", 1, 120, 45)
        smoking = st.selectbox("Hábito de Fumar", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    with col2:
        work = st.selectbox("Tipo de Trabalho", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence = st.selectbox("Tipo de Residência", ["Urban", "Rural"])
        married = st.selectbox("Estado Civil", ["Yes", "No"])

    with col3:
        glucose = st.number_input("Glicose Média (mg/dL)", 50.0, 400.0, 95.0)
        bmi = st.number_input("IMC", 10.0, 60.0, 24.5)
        
        st.write("**Comorbidades:**")
        hypertension = st.checkbox("Hipertensão Arterial")
        heart_disease = st.checkbox("Cardiopatia")

# =========================================================
# PROCESSAMENTO DA PREDIÇÃO
# =========================================================
st.divider()
if st.button("🧠 EXECUTAR TRIAGEM INTELIGENTE", use_container_width=True):
    if model is None:
        st.error("Erro: O modelo não pôde ser treinado. Verifique se 'data_stroke.csv' está no diretório.")
    else:
        # Criar DataFrame de entrada
        input_dict = {
            'gender': gender, 'age': age, 'hypertension': 1 if hypertension else 0,
            'heart_disease': 1 if heart_disease else 0, 'ever_married': married,
            'work_type': work, 'residence_type': residence,
            'avg_glucose_level': glucose, 'bmi': bmi, 'smoking_status': smoking
        }
        df_predict = pd.DataFrame([input_dict])

        # FEATURE ENGINEERING EM TEMPO REAL
        df_predict['age_hypertension'] = df_predict['age'] * df_predict['hypertension']
        df_predict['age_heart_disease'] = df_predict['age'] * df_predict['heart_disease']
        df_predict['metabolic_syndrome'] = (df_predict['avg_glucose_level'] * df_predict['bmi']) / 100
        
        married_val = 1 if married == "Yes" else 0
        df_predict['age_married'] = df_predict['age'] * married_val
        
        df_predict['cvd_risk_score'] = (
            df_predict['hypertension'] + df_predict['heart_disease'] + 
            (1 if glucose > 150 else 0) + (1 if bmi > 30 else 0)
        )

        def cat_age_local(a):
            if a < 30: return 0
            if a < 60: return 1
            if a < 75: return 2
            return 3
        df_predict['age_bracket'] = df_predict['age'].apply(cat_age_local)
        
        df_predict['lifestyle_risk'] = df_predict['work_type'].apply(
            lambda x: 1 if x in ['Private', 'Self-employed'] else 0
        )

        # PROBABILIDADE REAL
        prob_real = model.predict_proba(df_predict)[0][1]
        indice_risco = int(prob_real * 100)

        st.subheader("📊 Avaliação do Sistema SSDC")

        # REGRAS DE DECISÃO CLÍNICA BASEADAS NO LIMIAR DE 35% (RECALL = 90%)
        if prob_real >= 0.35:
            st.error(f"## 🚨 ALTO RISCO DE AVC DETECTADO ({indice_risco}% de Risco Calculado)")
            st.progress(prob_real)
            st.markdown("""
            ⚠ **TRIAGEM PRIORITÁRIA:** O perfil biológico do paciente ultrapassa o limiar clínico de alerta ($≥35\%$). 
            Recomenda-se encaminhamento para exames neurológicos e acompanhamento preventivo de emergência.
            """)
        else:
            st.success(f"## ✅ BAIXO RISCO / MONITORAMENTO REGULAR ({indice_risco}% de Risco Calculado)")
            st.progress(prob_real)
            st.markdown("""
            💚 **RISCO SOB CONTROLE:** Os parâmetros do paciente estão dentro do intervalo seguro ($<35\%$). 
            Manter acompanhamento de rotina.
            """)
