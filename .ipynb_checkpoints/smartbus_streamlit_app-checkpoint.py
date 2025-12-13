import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="SmartBus Predictions",
    page_icon="🚍",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT DU MODÈLE (avec cache)
# =============================================================================

@st.cache_resource
def load_model():
    """Charger le modèle et les métadonnées (mise en cache)"""
    model = xgb.XGBRegressor()
    model.load_model('smartbus_xgboost_model.json')
    
    with open('smartbus_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return model, metadata

# Charger le modèle
try:
    model, metadata = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")
    model_loaded = False

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 class="main-header">🚍 SmartBus - Prédiction de passagers</h1>', 
            unsafe_allow_html=True)

if model_loaded:
    st.success(f"✅ Modèle chargé | Performance : RMSE={metadata['performance']['rmse_test']:.2f} | R²={metadata['performance']['r2_test']:.3f}")
else:
    st.stop()

# =============================================================================
# SIDEBAR - NAVIGATION
# =============================================================================

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Choisir une page :",
    ["📊 Prédiction unique", "📅 Analyse journalière", "📈 Comparaison hebdomadaire", "ℹ️ Informations"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 À propos")
st.sidebar.info("""
**SmartBus Predictor**  
Version 1.0  

Prédiction du nombre de passagers montant dans les bus en fonction de :
- Jour et heure
- Type d'horaire
- Nombre d'arrêts
""")

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def make_prediction(data_dict):
    """Faire une prédiction à partir d'un dictionnaire"""
    df = pd.DataFrame([data_dict])
    df = df[metadata['features']]
    
    for col in ['Day Week', 'Schedule Type']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    prediction = model.predict(df)[0]
    return prediction

def create_gauge_chart(value, title="Prédiction"):
    """Créer un graphique de jauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        number={'suffix': " passagers", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100000], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#2E86AB"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20000], 'color': '#E8F4F8'},
                {'range': [20000, 50000], 'color': '#B8E6F0'},
                {'range': [50000, 100000], 'color': '#88D8E8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

# =============================================================================
# PAGE 1 : PRÉDICTION UNIQUE
# =============================================================================

if page == "📊 Prédiction unique":
    st.header("📊 Prédiction pour une observation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📅 Date")
        date_input = st.date_input("Date", datetime.now())
        annee = date_input.year
        mois = date_input.month
        jour = date_input.day
        
        day_week = st.selectbox(
            "Jour de la semaine",
            ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        )
        
        schedule_type = st.selectbox(
            "Type d'horaire",
            ["NORMAL", "SAMEDI", "DIMANCHE", "VACANCES"]
        )
    
    with col2:
        st.subheader("⏰ Heure")
        time_slot = st.slider("Créneau horaire (heure)", 0, 23, 8)
        
        week_index = st.number_input("Semaine de l'année", 1, 52, date_input.isocalendar()[1])
        
        day_week_map = {"Lundi": 0, "Mardi": 1, "Mercredi": 2, "Jeudi": 3, 
                        "Vendredi": 4, "Samedi": 5, "Dimanche": 6}
        index_day_week = day_week_map[day_week]
    
    with col3:
        st.subheader("🚏 Service")
        num_stops = st.number_input("Nombre d'arrêts", 0, 20000, 10000, step=100)
        
        st.info(f"""
        **Valeurs typiques :**
        - Heures de pointe : 12,000-16,000
        - Heures normales : 6,000-10,000
        - Nuit/Dimanche : 2,000-5,000
        """)
    
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
    
    # Bouton de prédiction
    if st.button("🔮 Faire la prédiction", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            prediction = make_prediction(input_data)
            
            st.markdown("---")
            st.subheader("📈 Résultat")
            
            col_result1, col_result2 = st.columns([1, 2])
            
            with col_result1:
                st.metric(
                    label="Nombre de passagers prédit",
                    value=f"{prediction:,.0f}",
                    delta=f"±{metadata['performance']['erreur_relative_%']:.1f}%"
                )
                
                st.info(f"""
                **Contexte :**
                - {day_week} à {time_slot}h
                - {num_stops:,} arrêts
                - Type : {schedule_type}
                """)
            
            with col_result2:
                gauge = create_gauge_chart(prediction, "Passagers prévus")
                st.plotly_chart(gauge, use_container_width=True)

# =============================================================================
# PAGE 2 : ANALYSE JOURNALIÈRE
# =============================================================================

elif page == "📅 Analyse journalière":
    st.header("📅 Analyse pour une journée complète")
    
    col1, col2 = st.columns(2)
    
    with col1:
        date_input = st.date_input("Date", datetime.now())
        day_week = st.selectbox(
            "Jour de la semaine",
            ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],
            key="day_analysis"
        )
    
    with col2:
        schedule_type = st.selectbox(
            "Type d'horaire",
            ["NORMAL", "SAMEDI", "DIMANCHE", "VACANCES"],
            key="schedule_analysis"
        )
    
    if st.button("📊 Analyser la journée", type="primary"):
        with st.spinner("Analyse en cours..."):
            # Créer les prédictions pour 24h
            hourly_predictions = []
            day_week_map = {"Lundi": 0, "Mardi": 1, "Mercredi": 2, "Jeudi": 3, 
                            "Vendredi": 4, "Samedi": 5, "Dimanche": 6}
            
            for hour in range(24):
                # Estimer le nombre d'arrêts
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    num_stops = 14000
                elif 0 <= hour <= 5:
                    num_stops = 2000
                else:
                    num_stops = 8000
                
                input_data = {
                    'Annee': date_input.year,
                    'Jour': date_input.day,
                    'Mois': date_input.month,
                    'Day Week': day_week,
                    'Schedule Type': schedule_type,
                    'Week Index': date_input.isocalendar()[1],
                    'Index Day Week': day_week_map[day_week],
                    'Time Slot': hour,
                    'Number of Stops': num_stops
                }
                
                pred = make_prediction(input_data)
                hourly_predictions.append({
                    'Heure': hour,
                    'Passagers': pred,
                    'Arrêts': num_stops
                })
            
            df_day = pd.DataFrame(hourly_predictions)
            
            # Métriques
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total journée", f"{df_day['Passagers'].sum():,.0f}")
            with col2:
                st.metric("📈 Moyenne horaire", f"{df_day['Passagers'].mean():,.0f}")
            with col3:
                peak_hour = df_day.loc[df_day['Passagers'].idxmax(), 'Heure']
                st.metric("⏰ Heure de pointe", f"{int(peak_hour)}h")
            with col4:
                st.metric("🔝 Max passagers", f"{df_day['Passagers'].max():,.0f}")
            
            # Graphique
            st.markdown("---")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_day['Heure'],
                y=df_day['Passagers'],
                mode='lines+markers',
                name='Passagers',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 171, 0.2)'
            ))
            
            fig.update_layout(
                title=f"Prédictions horaires - {day_week} {date_input}",
                xaxis_title="Heure de la journée",
                yaxis_title="Nombre de passagers",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé
            st.markdown("---")
            st.subheader("📋 Détails horaires")
            st.dataframe(
                df_day.style.format({
                    'Passagers': '{:,.0f}',
                    'Arrêts': '{:,.0f}'
                }).background_gradient(subset=['Passagers'], cmap='Blues'),
                use_container_width=True
            )

# =============================================================================
# PAGE 3 : COMPARAISON HEBDOMADAIRE
# =============================================================================

elif page == "📈 Comparaison hebdomadaire":
    st.header("📈 Comparaison des jours de la semaine")
    
    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    schedules = ["NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL", "SAMEDI", "DIMANCHE"]
    
    if st.button("🔄 Générer la comparaison", type="primary"):
        with st.spinner("Génération des prédictions..."):
            all_predictions = []
            
            for day, schedule in zip(days, schedules):
                day_week_map = {"Lundi": 0, "Mardi": 1, "Mercredi": 2, "Jeudi": 3, 
                                "Vendredi": 4, "Samedi": 5, "Dimanche": 6}
                
                for hour in range(24):
                    if 7 <= hour <= 9 or 17 <= hour <= 19:
                        num_stops = 14000
                    elif 0 <= hour <= 5:
                        num_stops = 2000
                    else:
                        num_stops = 8000
                    
                    input_data = {
                        'Annee': 2024,
                        'Jour': 18,
                        'Mois': 12,
                        'Day Week': day,
                        'Schedule Type': schedule,
                        'Week Index': 50,
                        'Index Day Week': day_week_map[day],
                        'Time Slot': hour,
                        'Number of Stops': num_stops
                    }
                    
                    pred = make_prediction(input_data)
                    all_predictions.append({
                        'Jour': day,
                        'Heure': hour,
                        'Passagers': pred
                    })
            
            df_week = pd.DataFrame(all_predictions)
            
            # Statistiques par jour
            daily_stats = df_week.groupby('Jour')['Passagers'].agg(['sum', 'mean', 'max']).reset_index()
            daily_stats.columns = ['Jour', 'Total', 'Moyenne', 'Maximum']
            
            st.markdown("---")
            st.subheader("📊 Statistiques par jour")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(
                    daily_stats.style.format({
                        'Total': '{:,.0f}',
                        'Moyenne': '{:,.0f}',
                        'Maximum': '{:,.0f}'
                    }).background_gradient(subset=['Total'], cmap='YlOrRd'),
                    use_container_width=True
                )
            
            with col2:
                fig_bar = px.bar(
                    daily_stats,
                    x='Jour',
                    y='Total',
                    title="Total de passagers par jour",
                    color='Total',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Graphique de comparaison
            st.markdown("---")
            st.subheader("📈 Évolution horaire par jour")
            
            fig = go.Figure()
            
            for day in days:
                df_day = df_week[df_week['Jour'] == day]
                fig.add_trace(go.Scatter(
                    x=df_day['Heure'],
                    y=df_day['Passagers'],
                    mode='lines+markers',
                    name=day,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                xaxis_title="Heure de la journée",
                yaxis_title="Nombre de passagers",
                height=600,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 4 : INFORMATIONS
# =============================================================================

elif page == "ℹ️ Informations":
    st.header("ℹ️ Informations sur le modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Performance du modèle")
        perf = metadata['performance']
        st.metric("RMSE Test", f"{perf['rmse_test']:.2f} passagers")
        st.metric("R² Test", f"{perf['r2_test']:.4f}")
        st.metric("Erreur relative", f"{perf['erreur_relative_%']:.2f}%")
        
        st.subheader("📅 Informations")
        st.info(f"**Date de création :** {metadata['date_creation']}")
        st.info(f"**Algorithme :** XGBoost Regressor")
        st.info(f"**Nombre de features :** {len(metadata['features'])}")
    
    with col2:
        st.subheader("⚙️ Hyperparamètres optimaux")
        params = metadata['meilleurs_parametres']
        for param, value in params.items():
            st.code(f"{param}: {value}")
        
        st.subheader("📋 Features utilisées")
        for i, feat in enumerate(metadata['features'], 1):
            st.text(f"{i}. {feat}")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>SmartBus Predictor v1.0 | "
    "Développé avec Streamlit et XGBoost</div>",
    unsafe_allow_html=True
)