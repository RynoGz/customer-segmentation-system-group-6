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
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Customer Segmentation Engine", className="text-center my-4 fw-bold"))),
    
    dbc.Row([
        # LEFT COLUMN: User Inputs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("New Customer Data", className="mb-0")),
                dbc.CardBody([
                    dbc.Label("Age"),
                    dbc.Input(id="age-in", type="number", value=30),
                    
                    dbc.Label("Family Size", className="mt-2"),
                    dbc.Input(id="family-in", type="number", value=2),
                    
                    dbc.Label("Work Experience (Years)", className="mt-2"),
                    dbc.Input(id="work-in", type="number", value=5),
                    
                    dbc.Label("Gender", className="mt-2"),
                    dcc.Dropdown(id="gender-in", options=[{'label': 'Male', 'value': 1}, {'label': 'Female', 'value': 0}], value=0),
                    
                    dbc.Label("Ever Married?", className="mt-2"),
                    dcc.Dropdown(id="married-in", options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=0),
                    
                    dbc.Label("Graduated?", className="mt-2"),
                    dcc.Dropdown(id="graduated-in", options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}], value=1),
                    
                    dbc.Label("Spending Score", className="mt-2"),
                    dcc.Dropdown(id="spending-in", options=[{'label': 'Low', 'value': 0}, {'label': 'Average', 'value': 1}, {'label': 'High', 'value': 2}], value=1),
                    
                    dbc.Label("Profession", className="mt-2"),
                    dcc.Dropdown(id="profession-in", options=[{'label': p, 'value': p} for p in ['Artist', 'Doctor', 'Engineer', 'Entertainment', 'Executive', 'Healthcare', 'Homemaker', 'Lawyer', 'Marketing']], value='Artist'),
                    
                    # Note the custom CSS class 'predict-btn' here
                    dbc.Button("GENERATE SEGMENT", id="predict-btn", color="primary", className="w-100 mt-4 predict-btn")
                ])
            ], className="custom-card") # Note the custom CSS class here
        ], md=4),
        
        # RIGHT COLUMN: Graph and Results
        dbc.Col([
            html.Div(id="result-display", className="mb-4"),
            dbc.Card([
                dbc.CardBody(dcc.Graph(id="cluster-graph"))
            ], className="custom-card shadow-sm")
        ], md=8)
    ])
], fluid=True, className="p-4")

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
    fig.update_layout(plot_bgcolor='white', transition_duration=500)
    
    if not n_clicks:
        return html.Div(), fig
        
    try:
        # 1. Build the exact 16-feature dictionary the Scaler expects
        input_data = {
            'Age': age, 'Work_Experience': work, 'Family_Size': family,
            'Gender_encoded': gender, 'Ever_Married_encoded': married,
            'Graduated_encoded': grad, 'Spending_encoded': spending,
            'Profession_Artist': 0, 'Profession_Doctor': 0, 'Profession_Engineer': 0,
            'Profession_Entertainment': 0, 'Profession_Executive': 0, 
            'Profession_Healthcare': 0, 'Profession_Homemaker': 0, 
            'Profession_Lawyer': 0, 'Profession_Marketing': 0
        }
        
        # 2. Inject the One-Hot Encoding for the chosen profession
        prof_key = f"Profession_{profession}"
        if prof_key in input_data:
            input_data[prof_key] = 1
            
        input_df = pd.DataFrame([input_data])
        
        # 3. Scale and Predict
        scaled_input = scaler.transform(input_df)
        cluster_idx = model.predict(scaled_input)[0]
        persona_name = persona_map.get(cluster_idx, f"Cluster {cluster_idx}")
        
        # 4. Highlight the predicted cluster on the graph
        df['Highlight'] = df['Cluster'].apply(lambda x: persona_name if x == cluster_idx else "Other")
        
        # Update figure to emphasize the selected cluster
        fig = px.scatter(df, x="PC1", y="PC2", color="Highlight", 
                         color_discrete_map={persona_name: "#e74c3c", "Other": "#bdc3c7"},
                         opacity=0.8, title=f"Prediction: {persona_name}")
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
        fig.update_layout(plot_bgcolor='white')
        
        # 5. Build the UI result badge
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