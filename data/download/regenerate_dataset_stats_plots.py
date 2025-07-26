import os
import fiftyone as fo
from fiftyone import ViewField as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
import json

print("🔥 REGENERATING DATASET FINAL STATS PLOTS (V2)")
print("="*80)

# --- Plotting Enhancements ---
font_title = {'fontsize': 22, 'fontweight': 'bold'}
font_label = {'fontsize': 18, 'fontweight': 'bold'}
font_tick = {'fontsize': 14}
font_legend = {'fontsize': 14}
font_text_box = {'fontsize': 14}
font_bar_label = {'fontsize': 12, 'fontweight': 'bold'}
font_pie_label = {'fontsize': 14, 'fontweight': 'bold'}

# Create output directory for plots
output_dir = "dataset_final_stats_v2"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Output directory: {output_dir}/")

# Load the dataset
try:
    ds = fo.load_dataset("met_textiles_27k")
    print(f"📊 FiftyOne dataset '{ds.name}' loaded with {len(ds)} samples.")
except ValueError:
    print("❌ FiftyOne dataset 'met_textiles_27k' not found.")
    print("Please ensure the dataset is available before running this script.")
    exit()


# Get tag counts
tag_counts = ds.count_sample_tags()
n_bad = tag_counts.get("bad", 0)
n_noise = tag_counts.get("noise", 0)
total_samples = len(ds)
clean_samples = total_samples - n_bad - n_noise

print("🎨 Creating comprehensive dataset visualizations with enhanced text...")

# 1. DATASET OVERVIEW PIE CHART
fig, ax = plt.subplots(figsize=(12, 10))
sizes = [clean_samples, n_bad, n_noise]
labels = [f'Clean Samples\n({count:,})' for count in sizes]
colors = ['#2ecc71', '#e74c3c', '#f39c12']
explode = (0.05, 0.05, 0.05)
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                  explode=explode, startangle=90, 
                                  textprops=font_pie_label,
                                  wedgeprops={'edgecolor': 'black', 'linewidth': 1})
ax.set_title('MET Textiles Dataset - Quality Distribution', **font_title, pad=20)
plt.tight_layout()
plt.savefig(f"{output_dir}/01_dataset_quality_overview.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 01_dataset_quality_overview.png")

# 2. CLEANING PROGRESS BAR CHART
fig, ax = plt.subplots(figsize=(14, 8))
categories = ['Total Dataset', 'Clean Samples', 'Removed (Bad)', 'Removed (Noise)']
values = [total_samples, clean_samples, n_bad, n_noise]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{value:,}\n({value/total_samples*100:.1f}%)',
            ha='center', va='bottom', **font_bar_label, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))
ax.set_ylabel('Number of Samples', **font_label)
ax.set_title('Dataset Cleaning Results', **font_title, pad=20)
ax.tick_params(axis='x', labelsize=font_tick['fontsize'])
ax.tick_params(axis='y', labelsize=font_tick['fontsize'])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(values) * 1.2)
plt.tight_layout()
plt.savefig(f"{output_dir}/02_cleaning_results.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 02_cleaning_results.png")

# 3. DEPARTMENT DISTRIBUTION (Clean samples only)
clean_view = ds.match(~F("tags").contains("bad")).match(~F("tags").contains("noise"))
if len(clean_view) > 0:
    dept_data = [s.department for s in clean_view if s.department]
    dept_counts = pd.Series(dept_data).value_counts()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    dept_counts.plot(kind='bar', ax=ax, color='#3498db', alpha=0.8, edgecolor='black')
    ax.set_title('Department Distribution (Clean Samples)', **font_title, pad=20)
    ax.set_xlabel('Department', **font_label)
    ax.set_ylabel('Number of Samples', **font_label)
    ax.tick_params(axis='x', rotation=45, ha='right', labelsize=font_tick['fontsize'])
    ax.tick_params(axis='y', labelsize=font_tick['fontsize'])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(dept_counts.values):
        ax.text(i, v, f'{v:,}', ha='center', va='bottom', **font_bar_label)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_departments_clean.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 03_departments_clean.png")

# 4. CLASSIFICATION DISTRIBUTION (Clean samples only)
if len(clean_view) > 0:
    class_data = [s.classification for s in clean_view if s.classification]
    class_counts = pd.Series(class_data).value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    class_counts.plot(kind='barh', ax=ax, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.set_title('Top 15 Classifications (Clean Samples)', **font_title, pad=20)
    ax.set_xlabel('Number of Samples', **font_label)
    ax.set_ylabel('Classification', **font_label)
    ax.tick_params(axis='x', labelsize=font_tick['fontsize'])
    ax.tick_params(axis='y', labelsize=font_tick['fontsize'])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    for i, v in enumerate(class_counts.values):
        ax.text(v, i, f' {v:,}', va='center', ha='left', **font_bar_label)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_classifications_clean.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 04_classifications_clean.png")

# 5. ALL TAGS DISTRIBUTION
if tag_counts:
    fig, ax = plt.subplots(figsize=(16, 10))
    tags_df = pd.Series(tag_counts).sort_values(ascending=False)
    colors = ['#e74c3c' if tag in ['bad', 'noise'] else '#3498db' for tag in tags_df.index]
    bars = ax.bar(tags_df.index, tags_df.values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Number of Samples', **font_label)
    ax.set_title('All Tags Distribution in Dataset', **font_title, pad=20)
    ax.tick_params(axis='x', rotation=45, ha='right', labelsize=font_tick['fontsize'])
    ax.tick_params(axis='y', labelsize=font_tick['fontsize'])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', **font_bar_label)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_all_tags_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 05_all_tags_distribution.png")

# 6. BEFORE vs AFTER COMPARISON
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
# Before
ax1.bar(['Original Dataset'], [total_samples], color='#95a5a6', alpha=0.8, width=0.5, edgecolor='black')
ax1.set_title('Before Cleaning', **font_title)
ax1.set_ylabel('Number of Samples', **font_label)
ax1.text(0, total_samples, f'{total_samples:,}\nsamples', ha='center', va='bottom', **font_bar_label)
ax1.set_ylim(0, total_samples * 1.1)
ax1.tick_params(axis='x', labelsize=font_tick['fontsize'])
ax1.tick_params(axis='y', labelsize=font_tick['fontsize'])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
# After
after_data = [clean_samples, n_bad + n_noise]
after_labels = ['Clean Samples', 'Removed\n(Bad + Noise)']
bars2 = ax2.bar(after_labels, after_data, color=['#2ecc71', '#e74c3c'], alpha=0.8, width=0.5, edgecolor='black')
ax2.set_title('After Cleaning', **font_title)
ax2.set_ylim(0, total_samples * 1.1)
ax2.tick_params(axis='x', labelsize=font_tick['fontsize'])
ax2.tick_params(axis='y', labelsize=font_tick['fontsize'])
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for bar, value in zip(bars2, after_data):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{value:,}\n({value/total_samples*100:.1f}%)', ha='center', va='bottom', **font_bar_label)
plt.suptitle('Dataset Cleaning: Before vs After', fontsize=font_title['fontsize'], fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{output_dir}/06_before_after_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 06_before_after_comparison.png")

# 7. SUMMARY STATISTICS TABLE
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')
summary_data = [
    ['Total Original Samples', f'{total_samples:,}', '100.0%'],
    ['Clean Samples', f'{clean_samples:,}', f'{clean_samples/total_samples*100:.1f}%'],
    ['Bad Samples (Removed)', f'{n_bad:,}', f'{n_bad/total_samples*100:.1f}%'],
    ['Noise Samples (Removed)', f'{n_noise:,}', f'{n_noise/total_samples*100:.1f}%'],
    ['Total Removed', f'{n_bad + n_noise:,}', f'{(n_bad + n_noise)/total_samples*100:.1f}%'],
    ['Data Retention Rate', f'{clean_samples:,}', f'{clean_samples/total_samples*100:.1f}%'],
]
if len(clean_view) > 0:
    unique_depts = len(set(s.department for s in clean_view if s.department))
    unique_classes = len(set(s.classification for s in clean_view if s.classification))
    summary_data.extend([
        ['Unique Departments (Clean)', f'{unique_depts}', ''],
        ['Unique Classifications (Clean)', f'{unique_classes}', ''],
    ])
table = ax.table(cellText=summary_data, colLabels=['Metric', 'Count', 'Percentage'], cellLoc='center', loc='center', colWidths=[0.5, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(1.2, 2.5)
for (i, j), cell in table.get_celld().items():
    cell.set_text_props(fontweight='bold')
    if i == 0:
        cell.set_facecolor('#3498db')
        cell.set_text_props(color='white')
    elif 'Clean' in summary_data[i-1][0] or 'Retention' in summary_data[i-1][0]:
        cell.set_facecolor('#d5f4e6')
    elif 'Removed' in summary_data[i-1][0] or 'Bad' in summary_data[i-1][0] or 'Noise' in summary_data[i-1][0]:
        cell.set_facecolor('#fadbd8')
ax.set_title('Final Statistics Summary', **font_title, pad=40)
plt.tight_layout()
plt.savefig(f"{output_dir}/07_summary_statistics.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 07_summary_statistics.png")

# 8. COMPREHENSIVE DASHBOARD
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
# Quality Score
ax1 = fig.add_subplot(gs[0, 0])
quality_score = clean_samples / total_samples * 100
ax1.pie([quality_score, 100-quality_score], colors=[('#2ecc71' if quality_score > 70 else '#f39c12'), '#ecf0f1'], startangle=90, counterclock=False, wedgeprops={'edgecolor': 'black', 'linewidth': 1})
ax1.add_artist(plt.Circle((0,0), 0.6, color='white'))
ax1.text(0, 0, f'{quality_score:.1f}%\nQuality', ha='center', va='center', fontsize=20, fontweight='bold')
ax1.set_title('Data Quality Score', **font_title)
# Retention Rate
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(['Data'], [clean_samples], color='#2ecc71', label='Retained', alpha=0.8, edgecolor='black')
ax2.bar(['Data'], [total_samples - clean_samples], bottom=[clean_samples], color='#e74c3c', label='Removed', alpha=0.8, edgecolor='black')
ax2.set_ylim(0, total_samples)
ax2.set_ylabel('Samples', **font_label)
ax2.set_title('Data Retention', **font_title)
ax2.text(0, clean_samples/2, f'Retained\n{clean_samples/total_samples*100:.1f}%', ha='center', va='center', **font_bar_label, color='white')
ax2.text(0, clean_samples + (total_samples-clean_samples)/2, f'Removed\n{(total_samples-clean_samples)/total_samples*100:.1f}%', ha='center', va='center', **font_bar_label, color='white')
ax2.tick_params(axis='x', labelsize=font_tick['fontsize'])
ax2.tick_params(axis='y', labelsize=font_tick['fontsize'])
# Department Dist
ax3 = fig.add_subplot(gs[1, :])
if 'dept_counts' in locals():
    top_depts = dept_counts.head(10)
    top_depts.plot(kind='bar', ax=ax3, color='#3498db', alpha=0.8, edgecolor='black')
    ax3.set_title('Top 10 Departments (Clean Data)', **font_title)
    ax3.set_xlabel('Department', **font_label)
    ax3.set_ylabel('Samples', **font_label)
    ax3.tick_params(axis='x', rotation=45, ha='right', labelsize=font_tick['fontsize'])
    ax3.tick_params(axis='y', labelsize=font_tick['fontsize'])
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
# Classification Dist
ax4 = fig.add_subplot(gs[2, :])
if 'class_counts' in locals():
    top_classes = class_counts.head(10)
    top_classes.plot(kind='barh', ax=ax4, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax4.set_title('Top 10 Classifications (Clean Data)', **font_title)
    ax4.set_xlabel('Samples', **font_label)
    ax4.set_ylabel('Classification', **font_label)
    ax4.tick_params(axis='x', labelsize=font_tick['fontsize'])
    ax4.tick_params(axis='y', labelsize=font_tick['fontsize'])
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    ax4.invert_yaxis()
plt.suptitle('MET Textiles Dataset - Comprehensive Final Report', fontsize=28, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{output_dir}/08_comprehensive_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 08_comprehensive_dashboard.png")

print("\n🎉 ALL PLOTS REGENERATED SUCCESSFULLY!")
print(f"✨ Directory: {output_dir}/")
