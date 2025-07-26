import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("🎨 RE-CREATING TEXMET FINAL DATASET IMAGE STATISTICS (V2)")
print("="*80)

# Define paths
base_dir = "/home/user1/Desktop/HAMZA/THESIS/TEXMET"
clean_dataset_dir = os.path.join(base_dir, "clean_dataset")

# Load the clean dataset
clean_json_path = os.path.join(clean_dataset_dir, "clean_textiles_dataset.json")
with open(clean_json_path, "r", encoding="utf-8") as f:
    clean_data = json.load(f)

df_clean = pd.DataFrame(clean_data)
print(f"📊 TeXMET Final dataset loaded: {len(df_clean):,} records")

# Create output directory for image statistics
output_dir = "texmet_final_image_stats_v2"
os.makedirs(output_dir, exist_ok=True)

# Get image file counts
def count_images_in_dir(directory):
    """Count image files in directory and return detailed stats"""
    if not os.path.exists(directory):
        return 0, []
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(directory).glob(f"*{ext}")))
        image_files.extend(list(Path(directory).glob(f"*{ext.upper()}")))
    
    return len(image_files), image_files

def get_image_object_ids_with_counts(directory):
    """Extract object IDs from image filenames and count images per object"""
    if not os.path.exists(directory):
        return {}, 0
    
    object_image_counts = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    total_files = 0
    
    for ext in image_extensions:
        for img_file in Path(directory).glob(f"*{ext}"):
            obj_id = img_file.name.split('_')[0]
            object_image_counts[int(obj_id)] = object_image_counts.get(int(obj_id), 0) + 1
            total_files += 1
        for img_file in Path(directory).glob(f"*{ext.upper()}"):
            obj_id = img_file.name.split('_')[0]
            object_image_counts[int(obj_id)] = object_image_counts.get(int(obj_id), 0) + 1
            total_files += 1
    
    return object_image_counts, total_files

# Get image statistics
clean_main_count, clean_main_files = count_images_in_dir(os.path.join(clean_dataset_dir, "images"))
clean_additional_count, clean_additional_files = count_images_in_dir(os.path.join(clean_dataset_dir, "additional_images"))

clean_main_obj_counts, _ = get_image_object_ids_with_counts(os.path.join(clean_dataset_dir, "images"))
clean_additional_obj_counts, _ = get_image_object_ids_with_counts(os.path.join(clean_dataset_dir, "additional_images"))

# Combine main and additional image counts per object
all_image_counts = {}
for obj_id in set(list(clean_main_obj_counts.keys()) + list(clean_additional_obj_counts.keys())):
    main_count = clean_main_obj_counts.get(obj_id, 0)
    additional_count = clean_additional_obj_counts.get(obj_id, 0)
    all_image_counts[obj_id] = main_count + additional_count

print(f"📸 Image file statistics:")
print(f"   • Main images: {clean_main_count:,}")
print(f"   • Additional images: {clean_additional_count:,}")
print(f"   • Total images: {clean_main_count + clean_additional_count:,}")
print(f"   • Objects with images: {len(all_image_counts):,}")

# Create comprehensive image statistics visualizations
fig = plt.figure(figsize=(24, 28))
gs = fig.add_gridspec(6, 3, hspace=0.5, wspace=0.4)

# --- Plotting Enhancements ---
font_title = {'fontsize': 18, 'fontweight': 'bold'}
font_label = {'fontsize': 14, 'fontweight': 'bold'}
font_tick = {'fontsize': 12}
font_text = {'fontsize': 12, 'fontweight': 'bold'}

# 1. IMAGE TYPE DISTRIBUTION
ax1 = fig.add_subplot(gs[0, 0])
image_types = ['Main Images', 'Additional Images']
image_counts = [clean_main_count, clean_additional_count]
colors = ['#3498db', '#e74c3c']

bars = ax1.bar(image_types, image_counts, color=colors, alpha=0.8)
ax1.set_title('Image Type Distribution', fontdict=font_title)
ax1.set_ylabel('Number of Images', fontdict=font_label)
ax1.tick_params(axis='x', labelsize=font_tick['fontsize'])
ax1.tick_params(axis='y', labelsize=font_tick['fontsize'])


for bar, count in zip(bars, image_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 200,
             f'{count:,}\n({count/(clean_main_count + clean_additional_count)*100:.1f}%)',
             ha='center', va='bottom', fontdict=font_text)

# 2. IMAGE COVERAGE ANALYSIS
ax2 = fig.add_subplot(gs[0, 1])
total_objects = len(df_clean)
objects_with_images = len(all_image_counts)
objects_without_images = total_objects - objects_with_images

coverage_data = [objects_with_images, objects_without_images]
coverage_labels = ['With Images', 'Without Images']
coverage_colors = ['#2ecc71', '#95a5a6']

wedges, texts, autotexts = ax2.pie(coverage_data, labels=coverage_labels, colors=coverage_colors,
                                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': font_tick['fontsize'], 'fontweight': 'bold'})
ax2.set_title('Image Coverage', fontdict=font_title)

# 3. IMAGES PER OBJECT DISTRIBUTION
ax3 = fig.add_subplot(gs[0, 2])
images_per_object = list(all_image_counts.values())
# Handle outliers by capping the bins
bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, max(images_per_object) + 1]
bin_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']

hist_counts, _ = np.histogram(images_per_object, bins=bins)
bars = ax3.bar(bin_labels, hist_counts, color='#f39c12', alpha=0.8)
ax3.set_title('Images per Object', fontdict=font_title)
ax3.set_xlabel('Number of Images', fontdict=font_label)
ax3.set_ylabel('Number of Objects', fontdict=font_label)
ax3.tick_params(axis='x', labelsize=font_tick['fontsize'])
ax3.tick_params(axis='y', labelsize=font_tick['fontsize'])


for bar, count in zip(bars, hist_counts):
    if count > 0:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{count:,}', ha='center', va='bottom', fontdict=font_text)

# 4. DEPARTMENT VS IMAGE AVAILABILITY
ax4 = fig.add_subplot(gs[1, :])
dept_image_stats = []

for dept in df_clean['department'].dropna().unique():
    dept_objects = df_clean[df_clean['department'] == dept]
    total_dept_objects = len(dept_objects)
    
    dept_with_images = 0
    dept_total_images = 0
    
    for _, obj in dept_objects.iterrows():
        obj_id = obj['objectID']
        if obj_id in all_image_counts:
            dept_with_images += 1
            dept_total_images += all_image_counts[obj_id]
    
    coverage_pct = (dept_with_images / total_dept_objects * 100) if total_dept_objects > 0 else 0
    avg_images = (dept_total_images / dept_with_images) if dept_with_images > 0 else 0
    
    dept_image_stats.append({
        'department': dept,
        'total_objects': total_dept_objects,
        'objects_with_images': dept_with_images,
        'total_images': dept_total_images,
        'coverage_percentage': coverage_pct,
        'avg_images_per_object': avg_images
    })

dept_df = pd.DataFrame(dept_image_stats)
# Handle outliers by showing top 12 departments by total objects
dept_df = dept_df.sort_values('total_objects', ascending=False).head(12)

width = 0.8
x = np.arange(len(dept_df))

bars1 = ax4.bar(x, dept_df['objects_with_images'], width, label='With Images', color='#2ecc71', alpha=0.8)
bars2 = ax4.bar(x, dept_df['total_objects'] - dept_df['objects_with_images'], width,
                bottom=dept_df['objects_with_images'], label='Without Images', color='#e74c3c', alpha=0.8)

ax4.set_title('Image Coverage by Department (Top 12)', fontdict=font_title)
ax4.set_xlabel('Department', fontdict=font_label)
ax4.set_ylabel('Number of Objects', fontdict=font_label)
ax4.set_xticks(x)
ax4.set_xticklabels(dept_df['department'], rotation=45, ha='right', fontsize=font_tick['fontsize'])
ax4.legend(fontsize=font_tick['fontsize'])
ax4.tick_params(axis='y', labelsize=font_tick['fontsize'])


for i, (bar1, bar2, pct) in enumerate(zip(bars1, bars2, dept_df['coverage_percentage'])):
    total_height = bar1.get_height() + bar2.get_height()
    ax4.text(bar1.get_x() + bar1.get_width()/2., total_height + 20,
             f'{pct:.1f}%', ha='center', va='bottom', fontdict=font_text)

# ... (rest of the plotting code with similar modifications) ...

# 9. IMAGE STATISTICS SUMMARY TABLE
ax9 = fig.add_subplot(gs[4, :])
ax9.axis('tight')
ax9.axis('off')

summary_stats = [
    ['Total Objects in TeXMET Final', f'{len(df_clean):,}', '100.0%'],
    ['Objects with Images', f'{objects_with_images:,}', f'{objects_with_images/len(df_clean)*100:.1f}%'],
    ['Objects without Images', f'{objects_without_images:,}', f'{objects_without_images/len(df_clean)*100:.1f}%'],
    ['Total Images (All Types)', f'{clean_main_count + clean_additional_count:,}', ''],
    ['Main Images', f'{clean_main_count:,}', f'{clean_main_count/(clean_main_count + clean_additional_count)*100:.1f}%'],
    ['Additional Images', f'{clean_additional_count:,}', f'{clean_additional_count/(clean_main_count + clean_additional_count)*100:.1f}%'],
    ['Average Images per Object (with images)', f'{np.mean(images_per_object):.1f}', ''],
    ['Max Images per Object', f'{max(images_per_object)}', ''],
    ['Objects with Multiple Images', f'{sum(1 for x in images_per_object if x > 1):,}', 
     f'{sum(1 for x in images_per_object if x > 1)/len(images_per_object)*100:.1f}%'],
]

table = ax9.table(cellText=summary_stats,
                  colLabels=['Metric', 'Count', 'Percentage'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.5, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.2, 2.2)

for i in range(len(summary_stats) + 1):
    for j in range(3):
        cell = table[(i, j)]
        cell.set_text_props(fontweight='bold')
        if i == 0:
            cell.set_facecolor('#3498db')
            cell.set_text_props(color='white')
        elif i % 2 == 0:
            cell.set_facecolor('#f8f9fa')

ax9.set_title('TeXMET Final - Image Statistics Summary', fontdict=font_title, pad=20)


plt.suptitle('TeXMET Final Dataset - Comprehensive Image Statistics Analysis (V2)', 
             fontsize=24, fontweight='bold', y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(f"{output_dir}/texmet_final_comprehensive_image_stats_v2.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\n🎉 VISUALIZATION COMPLETE!")
print(f"📁 Plot saved to: {output_dir}/texmet_final_comprehensive_image_stats_v2.png")