import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="SmartBus Predictions",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT DU MODÈLE (avec cache)
# =============================================================================

@st.cache_resource
def load_model():
    """Charger le modèle LightGBM"""
    try:
        model = joblib.load('LightGBM_Optimisé.joblib')
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 class="main-header">SmartBus - Prédiction de passagers</h1>', unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("Navigation")
st.sidebar.info("""
SmartBus Predictor - Version 2.0  
Prédiction du nombre de passagers selon :
- Jour et heure
- Type d'horaire
- Nombre d'arrêts
""")

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def make_prediction(data_dict):
    features_order = [
        'Annee', 'Jour', 'Mois', 'Day Week', 'Schedule Type',
        'Week Index', 'Index Day Week', 'Time Slot', 'Number of Stops'
    ]
    df = pd.DataFrame([data_dict])[features_order]
    for col in ['Day Week', 'Schedule Type']:
        df[col] = df[col].astype('category')
    return max(0, model.predict(df)[0])

def create_gauge_chart(value, title="Prédiction"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        number={'suffix': " passagers", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100000]},
            'bar': {'color': "#2E86AB"},
            'steps': [
                {'range': [0, 20000], 'color': '#E8F4F8'},
                {'range': [20000, 50000], 'color': '#B8E6F0'},
                {'range': [50000, 100000], 'color': '#88D8E8'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': value}
        }
    ))
    fig.update_layout(height=400)
    return fig

# =============================================================================
# PAGE DE PRÉDICTION
# =============================================================================

st.header("Prédiction pour une observation")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Date")
    date_input = st.date_input("Date", datetime.now())
    annee, mois, jour = date_input.year, date_input.month, date_input.day

    day_week = st.selectbox("Jour de la semaine",
                            ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"])
    schedule_type = st.selectbox("Type d'horaire", ["NORMAL", "SAMEDI", "DIMANCHE", "VACANCES"])

with col2:
    st.subheader("Heure")
    time_slot = st.slider("Créneau horaire (heure)", 0, 23, 8)
    week_index = st.number_input("Semaine de l'année", 1, 52, date_input.isocalendar()[1])

    day_week_map = {"Lundi":0, "Mardi":1, "Mercredi":2, "Jeudi":3,
                    "Vendredi":4, "Samedi":5, "Dimanche":6}
    index_day_week = day_week_map[day_week]

with col3:
    st.subheader("Service")
    num_stops = st.number_input("Nombre d'arrêts", 0, 20000, 10000, step=100)

# Créer le dictionnaire de données
input_data = {
    'Annee': annee,
    'Jour': jour,
    'Mois': mois,
    'Day Week': day_week,
    'Schedule Type': schedule_type,
    'Week Index': week_index,
    'Index Day Week': index_day_week,
    'Time Slot': time_slot,
    'Number of Stops': num_stops
}

if st.button("Faire la prédiction"):
    try:
        prediction = make_prediction(input_data)
        st.markdown("---")
        st.subheader("Résultat")
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric("Nombre de passagers prédit", f"{prediction:,.0f}")
            st.info(f"**Contexte :**\n- {day_week} à {time_slot}h\n- {num_stops:,} arrêts\n- Type : {schedule_type}")
        with col_res2:
            st.plotly_chart(create_gauge_chart(prediction), use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")


st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>SmartBus Predictor v2.0</div>", unsafe_allow_html=True)

