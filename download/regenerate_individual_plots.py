import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

print("🔥 REGENERATING TEXMET FINAL - INDIVIDUAL PLOTS (V2)")
print("="*80)

# --- Plotting Enhancements ---
font_title = {'fontsize': 22, 'fontweight': 'bold'}
font_label = {'fontsize': 18, 'fontweight': 'bold'}
font_tick = {'fontsize': 14}
font_legend = {'fontsize': 14}
font_text_box = {'fontsize': 14}
font_bar_label = {'fontsize': 12, 'fontweight': 'bold'}

# Load clean dataset
base_dir = "/home/user1/Desktop/HAMZA/THESIS/TEXMET"
clean_dataset_dir = os.path.join(base_dir, "clean_dataset")

with open(os.path.join(clean_dataset_dir, "clean_textiles_dataset.json"), "r") as f:
    clean_data = json.load(f)

df_clean = pd.DataFrame(clean_data)
print(f"📊 Loaded: {len(df_clean):,} records")

# Create NEW output directory for FINAL RUN
output_dir = "TEXMET_FINAL_INDIVIDUAL_PLOTS_V2"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Output directory: {output_dir}/")

def analyze_ALL_image_characteristics(images_dir):
    if not os.path.exists(images_dir):
        print(f"❌ Directory not found: {images_dir}")
        return []
    
    image_data = []
    image_files = (list(Path(images_dir).glob("*.jpg")) + 
                  list(Path(images_dir).glob("*.jpeg")) + 
                  list(Path(images_dir).glob("*.png")) +
                  list(Path(images_dir).glob("*.JPG")) +
                  list(Path(images_dir).glob("*.JPEG")) +
                  list(Path(images_dir).glob("*.PNG")))
    
    print(f"🔍 Found {len(image_files)} images in {os.path.basename(images_dir)}")
    print(f"🚀 Analyzing ALL images...")
    
    for i, img_file in enumerate(image_files):
        try:
            file_size_mb = img_file.stat().st_size / (1024 * 1024)
            with Image.open(img_file) as img:
                width, height = img.size
                image_data.append({
                    'object_id': int(img_file.name.split('_')[0]),
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height if height > 0 else 0,
                    'file_size_mb': file_size_mb,
                    'total_pixels': width * height,
                    'megapixels': (width * height) / 1_000_000,
                    'format': img.format,
                    'mode': img.mode,
                })
        except Exception:
            continue
    print(f"🎉 Analysis complete! Processed {len(image_data)} images.")
    return image_data

print(f"\n📁 Analyzing ALL MAIN images...")
main_images_data = analyze_ALL_image_characteristics(os.path.join(clean_dataset_dir, "images"))
df_images = pd.DataFrame(main_images_data)
print(f"\n📈 Final dataset: {len(df_images):,} images analyzed")

if df_images.empty:
    print("❌ No images found! Exiting.")
    exit()

print(f"\n🎨 Creating 16 individual plots with enhanced visibility...")

# 1. IMAGE DIMENSIONS SCATTER
plt.figure(figsize=(16, 12))
scatter = plt.scatter(df_images['width'], df_images['height'], alpha=0.6, s=25, 
                     c=df_images['file_size_mb'], cmap='viridis_r', edgecolors='black', linewidth=0.1)
cbar = plt.colorbar(scatter)
cbar.set_label('File Size (MB)', **font_label)
cbar.ax.tick_params(labelsize=font_tick['fontsize'])
plt.xlabel('Width (pixels)', **font_label)
plt.ylabel('Height (pixels)', **font_label)
plt.title(f'Image Dimensions Distribution ({len(df_images):,} Images)', **font_title, pad=20)
plt.xticks(**font_tick)
plt.yticks(**font_tick)
plt.grid(True, alpha=0.3)
stats_text = f"Avg Dim: {df_images['width'].mean():.0f}x{df_images['height'].mean():.0f}\nAvg Size: {df_images['file_size_mb'].mean():.2f} MB\nAvg Res: {df_images['megapixels'].mean():.1f} MP"
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, **font_text_box,
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig(f"{output_dir}/01_dimensions_scatter_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 01_dimensions_scatter_plot.png")

# 2. FILE SIZE DISTRIBUTION
plt.figure(figsize=(14, 10))
plt.hist(df_images['file_size_mb'], bins=50, alpha=0.8, color='#e74c3c', edgecolor='black')
plt.xlabel('File Size (MB)', **font_label)
plt.ylabel('Number of Images', **font_label)
plt.title(f'File Size Distribution', **font_title, pad=20)
plt.axvline(df_images['file_size_mb'].mean(), color='blue', linestyle='--', linewidth=3, label=f"Mean: {df_images['file_size_mb'].mean():.2f} MB")
plt.axvline(df_images['file_size_mb'].median(), color='red', linestyle='--', linewidth=3, label=f"Median: {df_images['file_size_mb'].median():.2f} MB")
plt.legend(**font_legend)
plt.xticks(**font_tick)
plt.yticks(**font_tick)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/02_file_size_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 02_file_size_distribution.png")

# 3. ASPECT RATIO DISTRIBUTION
plt.figure(figsize=(14, 10))
# Filter out extreme outliers for better visualization
ar_filtered = df_images['aspect_ratio'][(df_images['aspect_ratio'] > 0.1) & (df_images['aspect_ratio'] < 4)]
plt.hist(ar_filtered, bins=60, alpha=0.8, color='#3498db', edgecolor='black')
plt.xlabel('Aspect Ratio (Width/Height)', **font_label)
plt.ylabel('Number of Images', **font_label)
plt.title('Aspect Ratio Distribution (Filtered)', **font_title, pad=20)
plt.axvline(1.0, color='red', linestyle='--', linewidth=3, label='Square (1:1)')
plt.axvline(4/3, color='orange', linestyle='--', linewidth=3, label='Classic TV (4:3)')
plt.axvline(16/9, color='green', linestyle='--', linewidth=3, label='Widescreen (16:9)')
plt.legend(**font_legend)
plt.xticks(**font_tick)
plt.yticks(**font_tick)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/03_aspect_ratio_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 03_aspect_ratio_distribution.png")

# 4. RESOLUTION CATEGORIES
plt.figure(figsize=(14, 10))
def categorize_resolution(px):
    if px < 5e5: return "Low (<0.5MP)"
    if px < 2e6: return "Medium (0.5-2MP)"
    if px < 8e6: return "High (2-8MP)"
    return "Very High (>8MP)"
df_images['resolution_category'] = df_images['total_pixels'].apply(categorize_resolution)
res_counts = df_images['resolution_category'].value_counts()
colors = ['#ff7f7f', '#ffbf7f', '#7fbf7f', '#7f7fff']
bars = plt.bar(res_counts.index, res_counts.values, color=colors, alpha=0.8, edgecolor='black')
plt.xlabel('Resolution Category', **font_label)
plt.ylabel('Number of Images', **font_label)
plt.title('Image Resolution Categories', **font_title, pad=20)
plt.xticks(rotation=45, **font_tick)
plt.yticks(**font_tick)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{int(yval):,}\n({yval/len(df_images)*100:.1f}%)', ha='center', va='bottom', **font_bar_label)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{output_dir}/04_resolution_categories.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 04_resolution_categories.png")

# 5. IMAGE FORMAT DISTRIBUTION
plt.figure(figsize=(12, 10))
format_counts = df_images['format'].value_counts()
labels = [f"{fmt} ({count:,})" for fmt, count in format_counts.items()]
plt.pie(format_counts.values, labels=labels, autopct='%1.1f%%', 
        startangle=90, textprops={'fontsize': font_legend['fontsize'], 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1})
plt.title('Image Format Distribution', **font_title, pad=20)
plt.tight_layout()
plt.savefig(f"{output_dir}/05_format_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 05_format_distribution.png")

# 6. MEGAPIXELS DISTRIBUTION
plt.figure(figsize=(14, 10))
plt.hist(df_images['megapixels'], bins=50, alpha=0.8, color='#9b59b6', edgecolor='black')
plt.xlabel('Megapixels (MP)', **font_label)
plt.ylabel('Number of Images', **font_label)
plt.title('Megapixels Distribution', **font_title, pad=20)
plt.axvline(df_images['megapixels'].mean(), color='red', linestyle='--', linewidth=3, label=f"Mean: {df_images['megapixels'].mean():.1f} MP")
plt.legend(**font_legend)
plt.xticks(**font_tick)
plt.yticks(**font_tick)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/06_megapixels_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 06_megapixels_distribution.png")

# 7. WIDTH VS HEIGHT CORRELATION
plt.figure(figsize=(14, 10))
correlation = df_images['width'].corr(df_images['height'])
plt.scatter(df_images['width'], df_images['height'], alpha=0.5, s=15, color='#1abc9c')
plt.xlabel('Width (pixels)', **font_label)
plt.ylabel('Height (pixels)', **font_label)
plt.title(f'Width vs Height Correlation (r = {correlation:.3f})', **font_title, pad=20)
max_dim = max(df_images['width'].max(), df_images['height'].max())
plt.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.7, linewidth=3, label='Square (1:1)')
plt.legend(**font_legend)
plt.xticks(**font_tick)
plt.yticks(**font_tick)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/07_width_height_correlation.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 07_width_height_correlation.png")

# 8. FILE SIZE VS RESOLUTION
plt.figure(figsize=(14, 10))
plt.scatter(df_images['megapixels'], df_images['file_size_mb'], alpha=0.5, s=15, color='#f39c12')
plt.xlabel('Megapixels (MP)', **font_label)
plt.ylabel('File Size (MB)', **font_label)
plt.title('File Size vs. Resolution', **font_title, pad=20)
size_res_corr = df_images['megapixels'].corr(df_images['file_size_mb'])
plt.text(0.05, 0.95, f'Correlation: {size_res_corr:.3f}', transform=plt.gca().transAxes,
         **font_text_box, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
plt.xticks(**font_tick)
plt.yticks(**font_tick)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/08_filesize_vs_resolution.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 08_filesize_vs_resolution.png")

# 9. SIZE CATEGORY BREAKDOWN
plt.figure(figsize=(14, 10))
def categorize_file_size(size_mb):
    if size_mb < 0.5: return "Tiny (<0.5MB)"
    if size_mb < 1.0: return "Small (0.5-1MB)"
    if size_mb < 2.0: return "Medium (1-2MB)"
    if size_mb < 5.0: return "Large (2-5MB)"
    return "Huge (>5MB)"
df_images['size_category'] = df_images['file_size_mb'].apply(categorize_file_size)
size_counts = df_images['size_category'].value_counts().reindex(["Tiny (<0.5MB)", "Small (0.5-1MB)", "Medium (1-2MB)", "Large (2-5MB)", "Huge (>5MB)"])
colors_size = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
bars = plt.bar(size_counts.index, size_counts.values, color=colors_size, alpha=0.8, edgecolor='black')
plt.xlabel('File Size Category', **font_label)
plt.ylabel('Number of Images', **font_label)
plt.title('File Size Categories', **font_title, pad=20)
plt.xticks(rotation=45, ha='right', **font_tick)
plt.yticks(**font_tick)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{int(yval):,}\n({yval/len(df_images)*100:.1f}%)', ha='center', va='bottom', **font_bar_label)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{output_dir}/09_size_categories.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 09_size_categories.png")

# 10. ASPECT RATIO CATEGORIES
plt.figure(figsize=(12, 10))
def categorize_aspect_ratio(r):
    if r < 0.8: return "Portrait (<0.8)"
    if r < 1.2: return "Square (0.8-1.2)"
    if r < 1.8: return "Landscape (1.2-1.8)"
    return "Wide (>1.8)"
df_images['aspect_category'] = df_images['aspect_ratio'].apply(categorize_aspect_ratio)
aspect_counts = df_images['aspect_category'].value_counts()
colors_aspect = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
plt.pie(aspect_counts.values, labels=aspect_counts.index, colors=colors_aspect, autopct='%1.1f%%', 
        startangle=90, textprops={'fontsize': font_legend['fontsize'], 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1})
plt.title('Aspect Ratio Categories', **font_title, pad=20)
plt.tight_layout()
plt.savefig(f"{output_dir}/10_aspect_ratio_categories.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 10_aspect_ratio_categories.png")

# 11. QUALITY METRICS DASHBOARD
plt.figure(figsize=(14, 10))
quality_metrics = {
    'High Res\n(>2MP)': len(df_images[df_images['megapixels'] > 2]) / len(df_images) * 100,
    'Good Size\n(>1MB)': len(df_images[df_images['file_size_mb'] > 1]) / len(df_images) * 100,
    'Standard\nFormat (JPG)': len(df_images[df_images['format'] == 'JPEG']) / len(df_images) * 100,
    'RGB Mode': len(df_images[df_images['mode'] == 'RGB']) / len(df_images) * 100
}
bars = plt.bar(quality_metrics.keys(), quality_metrics.values(), color=['#2ecc71', '#3498db', '#f39c12', '#9b59b6'], alpha=0.8)
plt.ylabel('Percentage of Dataset (%)', **font_label)
plt.title('Image Quality Metrics', **font_title, pad=20)
plt.ylim(0, 100)
plt.xticks(**font_tick)
plt.yticks(**font_tick)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', **font_bar_label)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{output_dir}/11_quality_metrics.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 11_quality_metrics.png")

# 12. TOP 10 IMAGES BY FILESIZE
plt.figure(figsize=(14, 10))
top_by_size = df_images.nlargest(10, 'file_size_mb')
bars = plt.barh(top_by_size['object_id'].astype(str), top_by_size['file_size_mb'], color='#e74c3c', alpha=0.8, edgecolor='black')
plt.xlabel('File Size (MB)', **font_label)
plt.ylabel('Object ID', **font_label)
plt.title('Top 10 Images by File Size', **font_title, pad=20)
plt.xticks(**font_tick)
plt.yticks(**font_tick)
plt.grid(True, alpha=0.3, axis='x')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{output_dir}/12_top_by_filesize.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 12_top_by_filesize.png")

# 13. TOP 10 IMAGES BY RESOLUTION
plt.figure(figsize=(14, 10))
top_by_res = df_images.nlargest(10, 'megapixels')
bars = plt.barh(top_by_res['object_id'].astype(str), top_by_res['megapixels'], color='#2ecc71', alpha=0.8, edgecolor='black')
plt.xlabel('Megapixels (MP)', **font_label)
plt.ylabel('Object ID', **font_label)
plt.title('Top 10 Images by Resolution', **font_title, pad=20)
plt.xticks(**font_tick)
plt.yticks(**font_tick)
plt.grid(True, alpha=0.3, axis='x')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{output_dir}/13_top_by_resolution.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 13_top_by_resolution.png")

# 14. EXTREME ASPECT RATIOS
plt.figure(figsize=(14, 10))
extreme_ratios = pd.concat([df_images.nsmallest(5, 'aspect_ratio'), df_images.nlargest(5, 'aspect_ratio')])
colors = ['#3498db'] * 5 + ['#f39c12'] * 5
labels = [f"ID: {oid}, AR: {ar:.2f}" for oid, ar in zip(extreme_ratios['object_id'], extreme_ratios['aspect_ratio'])]
bars = plt.barh(labels, extreme_ratios['aspect_ratio'], color=colors, alpha=0.8, edgecolor='black')
plt.xlabel('Aspect Ratio', **font_label)
plt.title('Extreme Aspect Ratios (5 Tallest & 5 Widest)', **font_title, pad=20)
plt.xticks(**font_tick)
plt.yticks(fontsize=font_tick['fontsize'])
plt.grid(True, alpha=0.3, axis='x')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{output_dir}/14_extreme_aspect_ratios.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 14_extreme_aspect_ratios.png")

# 15. CORRELATION MATRIX
plt.figure(figsize=(12, 10))
numerical_cols = ['width', 'height', 'aspect_ratio', 'file_size_mb', 'megapixels']
correlation_matrix = df_images[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', fmt='.3f', 
            annot_kws={"size": font_legend['fontsize'], "weight": "bold"}, 
            linewidths=.5, vmin=-1, vmax=1)
plt.title('Image Characteristics Correlation Matrix', **font_title, pad=20)
plt.xticks(rotation=45, ha='right', **font_tick)
plt.yticks(rotation=0, **font_tick)
plt.tight_layout()
plt.savefig(f"{output_dir}/15_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 15_correlation_matrix.png")

# 16. COMPREHENSIVE STATISTICS TABLE
fig, ax = plt.subplots(figsize=(16, 12))
ax.axis('tight')
ax.axis('off')
stats_data = [
    ['Total Images Analyzed', f'{len(df_images):,}', '100%'],
    ['Unique Objects', f'{df_images["object_id"].nunique():,}', f'{df_images["object_id"].nunique()/len(df_images)*100:.1f}%'],
    ['Avg Width', f'{df_images["width"].mean():.0f} px', f'Range: {df_images["width"].min()}-{df_images["width"].max()}'],
    ['Avg Height', f'{df_images["height"].mean():.0f} px', f'Range: {df_images["height"].min()}-{df_images["height"].max()}'],
    ['Avg File Size', f'{df_images["file_size_mb"].mean():.2f} MB', f'Range: {df_images["file_size_mb"].min():.2f}-{df_images["file_size_mb"].max():.2f}'],
    ['Avg Aspect Ratio', f'{df_images["aspect_ratio"].mean():.2f}', f'Range: {df_images["aspect_ratio"].min():.2f}-{df_images["aspect_ratio"].max():.2f}'],
    ['Avg Resolution', f'{df_images["megapixels"].mean():.1f} MP', f'Range: {df_images["megapixels"].min():.1f}-{df_images["megapixels"].max():.1f}'],
    ['Most Common Format', f'{df_images["format"].mode().iloc[0]}', f'{df_images["format"].value_counts().iloc[0]:,} images'],
    ['Largest Image (by Res)', f'{df_images.loc[df_images["total_pixels"].idxmax(), "width"]:.0f}x{df_images.loc[df_images["total_pixels"].idxmax(), "height"]:.0f}', f'{df_images["megapixels"].max():.1f} MP'],
]
table = ax.table(cellText=stats_data, colLabels=['Metric', 'Value', 'Details'], cellLoc='center', loc='center', colWidths=[0.3, 0.25, 0.45])
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(1.2, 3.0)
for (i, j), cell in table.get_celld().items():
    cell.set_text_props(fontweight='bold')
    if i == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white')
    elif i % 2 == 1:
        cell.set_facecolor('#ecf0f1')
ax.set_title('TeXMET Final - Image Statistics Summary', **font_title, pad=40)
plt.tight_layout()
plt.savefig(f"{output_dir}/16_comprehensive_statistics_table.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 16_comprehensive_statistics_table.png")

print(f"\n🎉 ALL {len(os.listdir(output_dir))} PLOTS REGENERATED SUCCESSFULLY!")
print(f"✨ Directory: {output_dir}/")
print("🚀 Ready for thesis presentation!")
