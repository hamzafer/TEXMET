import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

print("🔥 REGENERATING TOP 5 DEPARTMENT GRID VISUALIZATION (V2)")
print("="*80)

# --- Plotting Enhancements ---
font_suptitle = {'fontsize': 28, 'fontweight': 'bold'}
font_title = {'fontsize': 18, 'fontweight': 'bold'}
font_no_image = {'fontsize': 16, 'fontweight': 'bold', 'color': 'red'}

# --- Data Loading ---
print("📊 Loading data...")
json_path = "../FINAL_CORRECTED_MET_TEXTILES_DATASET/objects_with_images_only/ALL_TEXTILES_AND_TAPESTRIES_WITH_IMAGES_20250705_230315.json"
try:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"✅ Data loaded successfully: {len(df)} records")
except FileNotFoundError:
    print(f"❌ ERROR: JSON file not found at {json_path}")
    exit()

# --- Directory Setup ---
images_dir = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/download/MET_TEXTILES_BULLETPROOF_DATASET/images"
output_dir = "category_visualizations_v2"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(images_dir):
    print(f"❌ ERROR: Images directory not found at {images_dir}")
    exit()

# --- Core Visualization Logic (Modified) ---
# Get the top 5 departments by object count
dept_counts = df['department'].value_counts()
top_departments = dept_counts.head(5).index.tolist()

print(f"🎨 Visualizing Top 5 Departments: {top_departments}")

n_depts = len(top_departments)
n_cols = 3  # Set a fixed number of columns for better layout
n_rows = math.ceil(n_depts / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
axes = axes.flatten()

for idx, dept in enumerate(top_departments):
    ax = axes[idx]
    subset = df[(df['department'] == dept) & (df['primaryImage'].apply(bool))]
    
    if not subset.empty:
        # Select a random sample to avoid showing the same image every time
        row = subset.sample(1).iloc[0]
        obj_id = str(row['objectID'])
        
        # Find the primary image file
        file_match = [f for f in os.listdir(images_dir) if f.startswith(f"{obj_id}_primary")]
        
        if file_match:
            img_path = os.path.join(images_dir, file_match[0])
            try:
                img = mpimg.imread(img_path)
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, "Image Error", ha='center', va='center', **font_no_image)
                print(f"Could not read {img_path}: {e}")
        else:
            ax.text(0.5, 0.5, "No Image File", ha='center', va='center', **font_no_image)
    else:
        ax.text(0.5, 0.5, "No Objects\nwith Images", ha='center', va='center', **font_no_image)
        
    # Set title with enhanced font
    ax.set_title(f"{dept}\n(Count: {dept_counts[dept]:,})", **font_title)
    ax.axis('off')

# Hide any unused subplots
for i in range(n_depts, len(axes)):
    axes[i].axis('off')

plt.suptitle('Top 5 Departments by Object Count', y=1.0, **font_suptitle)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
fig_path = os.path.join(output_dir, "departments_grid_top5_v2.png")
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: {fig_path}")
plt.close()

print("\n🎉 Script finished successfully!")
