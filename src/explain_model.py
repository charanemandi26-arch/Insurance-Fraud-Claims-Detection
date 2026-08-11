import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')

def generate_feature_importance():
    print("Loading model artifacts...")
    model_path = os.path.join('model', 'model.pkl')
    features_path = os.path.join('model', 'feature_names.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print("Model artifacts not found. Please run save_final_model.py first.")
        return
        
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    
    print("Extracting feature importances...")
    importances = model.feature_importances_
    
    # Create a DataFrame
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort and get top 10
    top_10 = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)
    
    # Clean up feature names for better readability
    def clean_name(name):
        name = name.replace('cat__', '').replace('num__', '')
        return name.replace('_', ' ').title()
        
    top_10['Feature'] = top_10['Feature'].apply(clean_name)
    
    print("Generating chart...")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_10, palette='viridis')
    
    plt.title('Top 10 Features Influencing Fraud Prediction', fontsize=14, pad=15)
    plt.xlabel('Relative Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    # Add a footnote clarifying that this does not prove fraud
    plt.figtext(0.01, 0.01, "*Note: These features influenced the model's prediction; they do not prove that fraud occurred.", 
                ha="left", fontsize=10, style='italic', color='gray')
                
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    os.makedirs(os.path.join('docs', 'figures'), exist_ok=True)
    save_path = os.path.join('docs', 'figures', 'feature_importance.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Feature importance chart saved to {save_path}")

if __name__ == '__main__':
    generate_feature_importance()
