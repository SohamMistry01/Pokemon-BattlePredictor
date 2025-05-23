import streamlit as st
import pandas as pd
import joblib
import warnings
import ast
import base64
warnings.simplefilter('ignore')

# Background Image CSS
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    bg_image_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{data}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(bg_image_style, unsafe_allow_html=True)

# Set Background
set_bg("pika.jpg")

# Load data and model
pokemon_df = pd.read_csv('final_pokemon.csv')
#pokemon_df['sprites'] = pokemon_df['sprites'].apply(lambda x: ast.literal_eval(x)['normal'])

def extract_animated_or_normal(sprite_str):
    try:
        sprite_dict = ast.literal_eval(sprite_str)
        return sprite_dict.get('animated') or sprite_dict.get('normal')
    except (ValueError, SyntaxError):
        return None

pokemon_df['sprites'] = pokemon_df['sprites'].apply(extract_animated_or_normal)

# Replace Type 2 'Normal' with Type 1
pokemon_df.loc[pokemon_df['Type 2'] == 'Normal', 'Type 2'] = pokemon_df.loc[pokemon_df['Type 2'] == 'Normal', 'Type 1']
model = joblib.load('xgb_pokemon_model.pkl')
trained_columns = joblib.load('xgb_feature_columns.pkl')

# Preprocess Total and type one-hot
pokemon_df['Total'] = pokemon_df['Attack'] + pokemon_df['Defense'] + pokemon_df['HP'] + \
                      pokemon_df['Sp. Atk'] + pokemon_df['Sp. Def'] + pokemon_df['Speed']
type1_dummies = pd.get_dummies(pokemon_df['Type 1'], prefix='Type1')
type2_dummies = pd.get_dummies(pokemon_df['Type 2'], prefix='Type2')
pokemon_df = pd.concat([pokemon_df, type1_dummies, type2_dummies], axis=1)
pokemon_df = pokemon_df.rename(columns={'#': 'Pokemon_ID'})

stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']
type_cols = [col for col in pokemon_df.columns if col.startswith('Type1_') or col.startswith('Type2_')]

def get_pokemon_features(name, prefix):
    row = pokemon_df[pokemon_df['Name'] == name]
    if row.empty:
        return None
    stat_values = row[['Total']].add_prefix(prefix + '_')
    type_values = row[type_cols].add_prefix(prefix + '_')
    all_cols = [prefix + '_Total'] + [prefix + '_' + col for col in type_cols]
    full_row = pd.DataFrame(columns=all_cols)

    for col in all_cols:
        if col in stat_values:
            full_row.at[0, col] = stat_values[col].values[0]
        elif col in type_values:
            full_row.at[0, col] = type_values[col].values[0]
        else:
            full_row.at[0, col] = 0
    return full_row

# --- Streamlit App ---
st.title("🔮 Pokémon Battle Predictor")
st.divider()

# Dropdowns for Pokémon selection
pokemon_names = sorted(pokemon_df['Name'].unique())
name1 = st.selectbox("Choose First Pokémon", pokemon_names)
name2 = st.selectbox("Choose Second Pokémon", pokemon_names)
st.divider()

def show_pokemon_info(name):
    row = pokemon_df[pokemon_df['Name'] == name][['Type 1', 'Type 2'] + stats + ['sprites']]
    if not row.empty:
        image_url = row['sprites'].values[0]
        st.markdown(
            f"<img src='{image_url}' alt='{name}' style='height:125px;'>",
            unsafe_allow_html=True
        )

        info_df = row.drop(columns=['sprites']).T.reset_index()
        info_df.columns = ['Attribute', 'Value']
        st.dataframe(info_df.set_index('Attribute'), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.info(f"📊 Stats: {name1}")
    show_pokemon_info(name1)
with col2:
    st.info(f"📊 Stats: {name2}")
    show_pokemon_info(name2)

# Prediction
st.divider()
if st.button("⚔️ Predict Winner"):
    first_poke = get_pokemon_features(name1, 'First')
    second_poke = get_pokemon_features(name2, 'Second')

    if first_poke is not None and second_poke is not None:
        sample_X = pd.concat([first_poke, second_poke], axis=1).fillna(0)
        sample_X = sample_X[trained_columns]  # Ensure correct column order
        pred = model.predict(sample_X)[0]
        winner = name1 if pred == 1 else name2
        st.success(f"🏆 **Predicted Winner: {winner}**")
