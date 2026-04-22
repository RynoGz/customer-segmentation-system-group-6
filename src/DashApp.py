import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import plotly.express as px

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'artifacts', 'kmeans_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'artifacts', 'scaler.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'artifacts', 'customers_segmented.csv')

# --- LOAD ARTIFACTS ---
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(DATA_PATH)

# Map clusters to personas created by your Data Analyst
persona_map = df.groupby('Cluster')['Persona'].first().to_dict()

# --- APP SETUP ---
# We use dbc.themes.FLATLY for a clean HTML layout, plus our custom CSS in the assets folder
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# --- LAYOUT (HTML Structure) ---
app.layout = html.Div([
    dbc.Container([
        dbc.Row(dbc.Col(html.H2("Customer Segmentation Engine", className="text-center mt-4 mb-3 fw-bold text-dark"))),
        
        dbc.Row([
            # LEFT COLUMN
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("New Customer Data", className="mb-0 text-dark fw-bold"), className="bg-transparent border-0 pt-3 pb-0"),
                    
                   
                    dbc.CardBody([
                        dbc.Label("Age", className="fw-bold text-dark mb-0", style={"fontSize": "0.9rem"}),
                        dbc.Input(id="age-in", type="number", value=30, size="sm", className="mb-2"),
                        
                        dbc.Label("Family Size", className="fw-bold text-dark mb-0", style={"fontSize": "0.9rem"}),
                        dbc.Input(id="family-in", type="number", value=2, size="sm", className="mb-2"),
                        
                        dbc.Label("Work Experience (Years)", className="fw-bold text-dark mb-0", style={"fontSize": "0.9rem"}),
                        dbc.Input(id="work-in", type="number", value=5, size="sm", className="mb-2"),
                        
                        dbc.Label("Gender", className="fw-bold text-dark mb-0", style={"fontSize": "0.9rem"}),
                        dcc.Dropdown(id="gender-in", options=[{'label': 'Male', 'value': 1}, {'label': 'Female', 'value': 0}], value=0, className="mb-2"),
                        
                        dbc.Label("Ever Married?", className="fw-bold text-dark mb-0", style={"fontSize": "0.9rem"}),
                        dcc.Dropdown(id="married-in", options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0, className="mb-2"),
                        
                        dbc.Label("Graduated?", className="fw-bold text-dark mb-0", style={"fontSize": "0.9rem"}),
                        dcc.Dropdown(id="graduated-in", options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=1, className="mb-2"),
                        
                        dbc.Label("Spending Score", className="fw-bold text-dark mb-0", style={"fontSize": "0.9rem"}),
                        dcc.Dropdown(id="spending-in", options=[{'label': 'Low', 'value': 0}, {'label': 'Average', 'value': 1}, {'label': 'High', 'value': 2}], value=1, className="mb-2"),
                        
                        dbc.Label("Profession", className="fw-bold text-dark mb-0", style={"fontSize": "0.9rem"}),
                        dcc.Dropdown(id="profession-in", options=[{'label': p, 'value': p} for p in ['Artist', 'Doctor', 'Engineer', 'Entertainment', 'Executive', 'Healthcare', 'Homemaker', 'Lawyer', 'Marketing']], value='Artist', className="mb-3"),
                        
                        dbc.Button("GENERATE SEGMENT", id="predict-btn", color="dark", className="w-100 predict-btn py-2")
                    ], className="px-3 py-2") 
                ], className="glass-card mb-4")
            ], md=4, lg=3),
            
            # RIGHT COLUMN
            dbc.Col([
                html.Div(id="result-display"),
                dbc.Card([
                    dbc.CardBody(dcc.Graph(id="cluster-graph", style={"height": "65vh", "minHeight": "450px"}))
                ], className="glass-card mb-4")
            ], md=8, lg=9)
        ])
    ], fluid=True, className="px-4") 
])

# --- ML INFERENCE LOGIC ---
@app.callback(
    [Output('result-display', 'children'), Output('cluster-graph', 'figure')],
    Input('predict-btn', 'n_clicks'),
    [State('age-in', 'value'), State('work-in', 'value'), State('family-in', 'value'),
     State('gender-in', 'value'), State('married-in', 'value'), State('graduated-in', 'value'),
     State('spending-in', 'value'), State('profession-in', 'value')]
)
def segment_customer(n_clicks, age, work, family, gender, married, grad, spending, profession):
    # Base Plot (Historical Data)
    fig = px.scatter(df, x="PC1", y="PC2", color="Persona", opacity=0.4, title="Customer Segments (PCA Projection)")
    fig.update_layout(
            paper_bgcolor='#ffffff', 
            plot_bgcolor='#ffffff',
            transition_duration=500,
            margin=dict(t=40, l=20, r=20, b=40)
        )
    
    if not n_clicks:
        return html.Div(), fig
        
    try:
        # Builds the exact 16-feature dictionary the Scaler expects
        input_data = {
            'Age': age, 'Work_Experience': work, 'Family_Size': family,
            'Gender_encoded': gender, 'Ever_Married_encoded': married,
            'Graduated_encoded': grad, 'Spending_encoded': spending,
            'Profession_Artist': 0, 'Profession_Doctor': 0, 'Profession_Engineer': 0,
            'Profession_Entertainment': 0, 'Profession_Executive': 0, 
            'Profession_Healthcare': 0, 'Profession_Homemaker': 0, 
            'Profession_Lawyer': 0, 'Profession_Marketing': 0
        }
        
        # Inject the One-Hot Encoding for the chosen profession
        prof_key = f"Profession_{profession}"
        if prof_key in input_data:
            input_data[prof_key] = 1
            
        input_df = pd.DataFrame([input_data])
        
        # Scale and Predict
        scaled_input = scaler.transform(input_df)
        cluster_idx = model.predict(scaled_input)[0]
        persona_name = persona_map.get(cluster_idx, f"Cluster {cluster_idx}")
        
        # Highlight the predicted cluster on the graph
        df['Highlight'] = df['Cluster'].apply(lambda x: persona_name if x == cluster_idx else "Other")
        
        # Update figure to emphasize the selected cluster
        fig = px.scatter(df, x="PC1", y="PC2", color="Highlight", 
                         color_discrete_map={persona_name: "#e74c3c", "Other": "#bdc3c7"},
                         opacity=0.8, title=f"Prediction: {persona_name}")
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
        fig.update_layout(
            paper_bgcolor='#ffffff', 
            plot_bgcolor='#ffffff',
            transition_duration=500,
            margin=dict(t=40, l=20, r=20, b=40)
        )
        
        # Build the UI result badge
        result_ui = dbc.Alert([
            html.H4("Analysis Complete", className="alert-heading"),
            html.Hr(),
            html.P("Based on the demographic and behavioural data, this customer is classified as:"),
            html.H3(f"{persona_name} (Cluster {cluster_idx})", className="fw-bold mb-0")
        ], color="success", className="custom-card")
        
        return result_ui, fig

    except Exception as e:
        return dbc.Alert(f"System Error: {str(e)}", color="danger"), fig

if __name__ == '__main__':
    app.run(debug=True)