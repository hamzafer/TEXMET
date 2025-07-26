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
font_title = {'fontsize': 24, 'fontweight': 'bold'}
font_label = {'fontsize': 20, 'fontweight': 'bold'}
font_tick = {'fontsize': 16}
font_legend = {'fontsize': 16}
font_text_box = {'fontsize': 16}
font_bar_label = {'fontsize': 14, 'fontweight': 'bold'}
font_pie_label = {'fontsize': 16, 'fontweight': 'bold'}

# Create output directory for plots
output_dir = "dataset_final_stats_v2"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Output directory: {output_dir}/")

# Load the FiftyOne dataset
try:
    ds = fo.load_dataset("met_textiles_27k")
    print(f"📊 FiftyOne dataset '{ds.name}' loaded with {len(ds)} samples.")
except ValueError:
    print("❌ FiftyOne dataset 'met_textiles_27k' not found. Exiting.")
    exit()

# --- Initial Data Preparation ---
tag_counts = ds.count_sample_tags()
n_bad = tag_counts.get("bad", 0)
n_noise = tag_counts.get("noise", 0)
total_samples = len(ds)
clean_samples = total_samples - n_bad - n_noise
clean_view = ds.match(~F("tags").contains("bad")).match(~F("tags").contains("noise"))

# Load original JSON data for detailed analysis
json_path = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/FINAL_CORRECTED_MET_TEXTILES_DATASET/objects_with_images_only/ALL_TEXTILES_AND_TAPESTRIES_WITH_IMAGES_20250705_230315.json"
with open(json_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)
df_original = pd.DataFrame(all_data)
clean_object_ids = {s.object_id for s in clean_view}
clean_df = df_original[df_original['objectID'].isin(clean_object_ids)].copy()
print(f"📊 Clean dataset loaded for detailed analysis: {len(clean_df):,} records")


print("\n🎨 Creating comprehensive dataset visualizations with enhanced text...")

# 1. DATASET OVERVIEW PIE CHART
fig, ax = plt.subplots(figsize=(12, 10))
sizes = [clean_samples, n_bad, n_noise]
label_names = ['Clean Samples', 'Bad Samples', 'Noise Samples']
labels = [f'{name}\n({count:,})' for name, count in zip(label_names, sizes)]
colors = ['#2ecc71', '#e74c3c', '#f39c12']
explode = (0.05, 0.05, 0.05)
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                  explode=explode, startangle=90,
                                  textprops=font_pie_label,
                                  wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
ax.set_title('MET Textiles Dataset - Quality Distribution', **font_title, pad=20)
plt.tight_layout()
plt.savefig(f"{output_dir}/01_dataset_quality_overview.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 01_dataset_quality_overview.png")

# 2. CLEANING PROGRESS BAR CHART
fig, ax = plt.subplots(figsize=(14, 10))
categories = ['Total Dataset', 'Clean Samples', 'Removed (Bad)', 'Removed (Noise)']
values = [total_samples, clean_samples, n_bad, n_noise]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = ax.bar(categories, values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{value:,}\n({value/total_samples*100:.1f}%)',
            ha='center', va='bottom', **font_bar_label, bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))
ax.set_ylabel('Number of Samples', **font_label)
ax.set_title('Dataset Cleaning Results', **font_title, pad=20)
ax.tick_params(axis='x', labelsize=font_tick['fontsize'])
ax.tick_params(axis='y', labelsize=font_tick['fontsize'])
ax.grid(axis='y', alpha=0.4, linestyle='--')
ax.set_ylim(0, max(values) * 1.25)
plt.tight_layout()
plt.savefig(f"{output_dir}/02_cleaning_results.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 02_cleaning_results.png")

# 3. DEPARTMENT DISTRIBUTION (Clean samples only)
if len(clean_view) > 0:
    dept_data = [s.department for s in clean_view if s.department]
    dept_counts = pd.Series(dept_data).value_counts()
    fig, ax = plt.subplots(figsize=(18, 12))
    dept_counts.plot(kind='bar', ax=ax, color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.set_title('Department Distribution (Clean Samples)', **font_title, pad=20)
    ax.set_xlabel('Department', **font_label)
    ax.set_ylabel('Number of Samples', **font_label)
    ax.tick_params(axis='x', labelsize=font_tick['fontsize'])
    ax.tick_params(axis='y', labelsize=font_tick['fontsize'])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.grid(axis='y', alpha=0.4, linestyle='--')
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
    fig, ax = plt.subplots(figsize=(18, 14))
    class_counts.plot(kind='barh', ax=ax, color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.set_title('Top 15 Classifications (Clean Samples)', **font_title, pad=20)
    ax.set_xlabel('Number of Samples', **font_label)
    ax.set_ylabel('Classification', **font_label)
    ax.tick_params(axis='x', labelsize=font_tick['fontsize'])
    ax.tick_params(axis='y', labelsize=font_tick['fontsize'])
    ax.grid(axis='x', alpha=0.4, linestyle='--')
    ax.invert_yaxis()
    for i, v in enumerate(class_counts.values):
        ax.text(v, i, f' {v:,}', va='center', ha='left', **font_bar_label)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_classifications_clean.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 04_classifications_clean.png")

# 5. ALL TAGS DISTRIBUTION
if tag_counts:
    fig, ax = plt.subplots(figsize=(18, 12))
    tags_df = pd.Series(tag_counts).sort_values(ascending=False)
    colors = ['#e74c3c' if tag in ['bad', 'noise'] else '#3498db' for tag in tags_df.index]
    bars = ax.bar(tags_df.index, tags_df.values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Samples', **font_label)
    ax.set_title('All Tags Distribution in Dataset', **font_title, pad=20)
    ax.tick_params(axis='x', labelsize=font_tick['fontsize'])
    ax.tick_params(axis='y', labelsize=font_tick['fontsize'])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', **font_bar_label)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_all_tags_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 05_all_tags_distribution.png")

# 6. BEFORE vs AFTER COMPARISON
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.bar(['Original Dataset'], [total_samples], color='#95a5a6', alpha=0.85, width=0.5, edgecolor='black', linewidth=1.5)
ax1.set_title('Before Cleaning', **font_title)
ax1.set_ylabel('Number of Samples', **font_label)
ax1.text(0, total_samples, f'{total_samples:,}\nsamples', ha='center', va='bottom', **font_bar_label)
ax1.set_ylim(0, total_samples * 1.15)
ax1.tick_params(axis='x', labelsize=font_tick['fontsize'])
ax1.tick_params(axis='y', labelsize=font_tick['fontsize'])
ax1.grid(axis='y', alpha=0.4, linestyle='--')
after_data = [clean_samples, n_bad + n_noise]
after_labels = ['Clean Samples', 'Removed\n(Bad + Noise)']
bars2 = ax2.bar(after_labels, after_data, color=['#2ecc71', '#e74c3c'], alpha=0.85, width=0.5, edgecolor='black', linewidth=1.5)
ax2.set_title('After Cleaning', **font_title)
ax2.set_ylim(0, total_samples * 1.15)
ax2.tick_params(axis='x', labelsize=font_tick['fontsize'])
ax2.tick_params(axis='y', labelsize=font_tick['fontsize'])
ax2.grid(axis='y', alpha=0.4, linestyle='--')
for bar, value in zip(bars2, after_data):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{value:,}\n({value/total_samples*100:.1f}%)', ha='center', va='bottom', **font_bar_label)
plt.suptitle('Dataset Cleaning: Before vs After', fontsize=font_title['fontsize'], fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(f"{output_dir}/06_before_after_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 06_before_after_comparison.png")

# 7. SUMMARY STATISTICS TABLE
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')
summary_data = [['Total Original Samples', f'{total_samples:,}', '100.0%'], ['Clean Samples', f'{clean_samples:,}', f'{clean_samples/total_samples*100:.1f}%'], ['Bad Samples (Removed)', f'{n_bad:,}', f'{n_bad/total_samples*100:.1f}%'], ['Noise Samples (Removed)', f'{n_noise:,}', f'{n_noise/total_samples*100:.1f}%'], ['Total Removed', f'{n_bad + n_noise:,}', f'{(n_bad + n_noise)/total_samples*100:.1f}%'], ['Data Retention Rate', f'{clean_samples:,}', f'{clean_samples/total_samples*100:.1f}%']]
if len(clean_view) > 0:
    unique_depts = len(set(s.department for s in clean_view if s.department))
    unique_classes = len(set(s.classification for s in clean_view if s.classification))
    summary_data.extend([['Unique Departments (Clean)', f'{unique_depts}', ''], ['Unique Classifications (Clean)', f'{unique_classes}', '']])
table = ax.table(cellText=summary_data, colLabels=['Metric', 'Count', 'Percentage'], cellLoc='center', loc='center', colWidths=[0.5, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(18)
table.scale(1.2, 3.0)
for (i, j), cell in table.get_celld().items():
    cell.set_text_props(fontweight='bold')
    if i == 0:
        cell.set_facecolor('#3498db')
        cell.set_text_props(color='white')
    elif 'Clean' in summary_data[i-1][0] or 'Retention' in summary_data[i-1][0]: cell.set_facecolor('#d5f4e6')
    elif 'Removed' in summary_data[i-1][0] or 'Bad' in summary_data[i-1][0] or 'Noise' in summary_data[i-1][0]: cell.set_facecolor('#fadbd8')
ax.set_title('Final Statistics Summary', **font_title, pad=40)
plt.tight_layout()
plt.savefig(f"{output_dir}/07_summary_statistics.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 07_summary_statistics.png")

# 8. COMPREHENSIVE DASHBOARD
fig = plt.figure(figsize=(22, 18))
gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
quality_score = clean_samples / total_samples * 100
ax1.pie([quality_score, 100-quality_score], colors=[('#2ecc71' if quality_score > 70 else '#f39c12'), '#ecf0f1'], startangle=90, counterclock=False, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
ax1.add_artist(plt.Circle((0,0), 0.65, color='white'))
ax1.text(0, 0, f'{quality_score:.1f}%\nQuality', ha='center', va='center', fontsize=24, fontweight='bold')
ax1.set_title('Data Quality Score', **font_title)
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(['Data'], [clean_samples], color='#2ecc71', label='Retained', alpha=0.85, edgecolor='black', linewidth=1.5)
ax2.bar(['Data'], [total_samples - clean_samples], bottom=[clean_samples], color='#e74c3c', label='Removed', alpha=0.85, edgecolor='black', linewidth=1.5)
ax2.set_ylim(0, total_samples)
ax2.set_ylabel('Samples', **font_label)
ax2.set_title('Data Retention', **font_title)
ax2.text(0, clean_samples/2, f'Retained\n{clean_samples/total_samples*100:.1f}%', ha='center', va='center', **font_bar_label, color='white')
ax2.text(0, clean_samples + (total_samples-clean_samples)/2, f'Removed\n{(total_samples-clean_samples)/total_samples*100:.1f}%', ha='center', va='center', **font_bar_label, color='white')
ax2.tick_params(axis='x', labelsize=font_tick['fontsize'])
ax2.tick_params(axis='y', labelsize=font_tick['fontsize'])
ax3 = fig.add_subplot(gs[1, :])
if 'dept_counts' in locals():
    top_depts = dept_counts.head(10)
    top_depts.plot(kind='bar', ax=ax3, color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    ax3.set_title('Top 10 Departments (Clean Data)', **font_title)
    ax3.set_xlabel('Department', **font_label)
    ax3.set_ylabel('Samples', **font_label)
    ax3.tick_params(axis='y', labelsize=font_tick['fontsize'])
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=font_tick['fontsize'])
    ax3.grid(axis='y', alpha=0.4, linestyle='--')
ax4 = fig.add_subplot(gs[2, :])
if 'class_counts' in locals():
    top_classes = class_counts.head(10)
    top_classes.plot(kind='barh', ax=ax4, color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    ax4.set_title('Top 10 Classifications (Clean Data)', **font_title)
    ax4.set_xlabel('Samples', **font_label)
    ax4.set_ylabel('Classification', **font_label)
    ax4.tick_params(axis='x', labelsize=font_tick['fontsize'])
    ax4.tick_params(axis='y', labelsize=font_tick['fontsize'])
    ax4.grid(axis='x', alpha=0.4, linestyle='--')
    ax4.invert_yaxis()
plt.suptitle('MET Textiles Dataset - Comprehensive Final Report', fontsize=32, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{output_dir}/08_comprehensive_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 08_comprehensive_dashboard.png")

# Plot 9: Comprehensive Department Analysis
dept_data_raw = [s.department for s in clean_view]
dept_data = [d if d and d.strip() else "Others" for d in dept_data_raw]
null_count = sum(1 for d in dept_data_raw if not d or not d.strip())
total_clean = len(dept_data_raw)
dept_counts = pd.Series(dept_data).value_counts()
fig = plt.figure(figsize=(22, 18))
gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.4)
ax1 = fig.add_subplot(gs[0, 0])
ax1.pie([total_clean - null_count, null_count], labels=['Valid', 'Missing'], colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90, textprops=font_pie_label, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
ax1.set_title('Department Data Quality', **font_title)
ax2 = fig.add_subplot(gs[0, 1])
top_5_sum = dept_counts.head(5).sum()
others_sum = dept_counts.sum() - top_5_sum
ax2.bar(['Top 5 Depts', 'All Others'], [top_5_sum, others_sum], color=['#2ecc71', '#95a5a6'], alpha=0.85, edgecolor='black', linewidth=1.5)
ax2.set_title('Top 5 vs. Others Distribution', **font_title)
ax2.set_ylabel('Number of Samples', **font_label)
ax2.tick_params(labelsize=font_tick['fontsize'])
ax3 = fig.add_subplot(gs[1, :])
top_depts = dept_counts.head(12)
top_depts.plot(kind='bar', ax=ax3, color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
ax3.set_title('Top 12 Departments (Clean Samples)', **font_title)
ax3.set_ylabel('Number of Samples', **font_label)
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=font_tick['fontsize'])
ax4 = fig.add_subplot(gs[2, :])
dept_counts.sort_values().plot(kind='barh', ax=ax4, color='#9b59b6', alpha=0.85, edgecolor='black', linewidth=1.5)
ax4.set_title('Complete Department Distribution', **font_title)
ax4.set_xlabel('Number of Samples', **font_label)
ax4.tick_params(labelsize=font_tick['fontsize'])
plt.suptitle('Comprehensive Department Analysis', fontsize=32, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{output_dir}/09_department_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 09_department_comprehensive_analysis.png")

# Plot 10: Dynamic Categories
fig, axes = plt.subplots(2, 2, figsize=(22, 18))
clean_df['objectBeginDate'] = pd.to_numeric(clean_df['objectBeginDate'], errors='coerce')
valid_dates = clean_df['objectBeginDate'].dropna()
if len(valid_dates) > 0:
    quantiles = np.linspace(0, 1, 9)
    period_boundaries = valid_dates.quantile(quantiles).astype(int).tolist()
    def categorize_period(date):
        if pd.isna(date): return "Unknown"
        for i in range(len(period_boundaries) - 1):
            if period_boundaries[i] <= date <= period_boundaries[i+1]: return f"{period_boundaries[i]}-{period_boundaries[i+1]}"
        return "Outside Range"
    clean_df['time_period'] = clean_df['objectBeginDate'].apply(categorize_period)
    period_counts = clean_df['time_period'].value_counts()
    period_counts.plot(kind='bar', ax=axes[0, 0], color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_title('Time Periods', **font_title)
    axes[0, 0].tick_params(axis='x', rotation=45, labelsize=font_tick['fontsize'])
all_mediums = clean_df['medium'].dropna().str.lower()
word_freq = Counter([w for med in all_mediums for w in med.replace(',', ' ').split() if len(w) > 3])
top_keywords = [w for w, c in word_freq.most_common(10)]
def categorize_medium(med):
    if pd.isna(med): return "Unknown"
    for k in top_keywords:
        if k in med.lower(): return f"{k.title()} Materials"
    return "Other"
clean_df['medium_category'] = clean_df['medium'].apply(categorize_medium)
medium_counts = clean_df['medium_category'].value_counts()
medium_counts.plot(kind='bar', ax=axes[0, 1], color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
axes[0, 1].set_title('Medium Categories', **font_title)
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=font_tick['fontsize'])
all_obj_words = Counter([w for name in clean_df['objectName'].dropna() for w in name.lower().split() if len(w) > 2])
common_obj_words = [w for w, c in all_obj_words.most_common(10)]
def categorize_object(name):
    if pd.isna(name): return "Unknown"
    for k in common_obj_words:
        if k in name.lower(): return f"{k.title()}-related"
    return "Other"
clean_df['object_category'] = clean_df['objectName'].apply(categorize_object)
obj_counts = clean_df['object_category'].value_counts()
obj_counts.plot(kind='barh', ax=axes[1, 0], color='#f39c12', alpha=0.85, edgecolor='black', linewidth=1.5)
axes[1, 0].set_title('Object Categories', **font_title)
axes[1, 0].tick_params(labelsize=font_tick['fontsize'])
clean_df['combined_geo'] = clean_df['country'].fillna('') + ' ' + clean_df['culture'].fillna('')
geo_words = Counter([w for geo in clean_df['combined_geo'] for w in geo.lower().split() if len(w) > 3])
common_geo_terms = [w for w, c in geo_words.most_common(10)]
def categorize_geo(geo):
    if pd.isna(geo) or not geo.strip(): return "Unknown"
    for t in common_geo_terms:
        if t in geo.lower(): return f"{t.title()} Region"
    return "Other"
clean_df['geographic_region'] = clean_df['combined_geo'].apply(categorize_geo)
geo_counts = clean_df['geographic_region'].value_counts()
geo_counts.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%', textprops=font_pie_label, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
axes[1, 1].set_title('Geographic Regions', **font_title)
axes[1, 1].set_ylabel('')
plt.suptitle('Dynamic Category Analysis (Clean Data)', fontsize=32, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{output_dir}/10_dynamic_categories_clean.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 10_dynamic_categories_clean.png")

# Plot 11: Time Periods Prettified
if 'period_counts' in locals():
    fig, ax = plt.subplots(figsize=(18, 12))
    colors = plt.cm.viridis_r(np.linspace(0.15, 0.85, len(period_counts)))
    bars = ax.bar(period_counts.index, period_counts.values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.9)
    ax.set_title('Time Periods Distribution (Clean Dataset)', **font_title, pad=24)
    ax.set_ylabel('Number of Samples', **font_label)
    ax.set_xlabel('Time Period', **font_label)
    ax.tick_params(axis='x', rotation=30, labelsize=font_tick['fontsize'])
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.tick_params(axis='y', labelsize=font_tick['fontsize'])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    total = period_counts.sum()
    for bar, value in zip(bars, period_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, f'{value:,}\n({value/total*100:.1f}%)', ha='center', va='bottom', **font_bar_label)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/11_time_periods_clean.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 11_time_periods_clean.png")

# Plot 12: Object Names Distribution
fig, ax = plt.subplots(figsize=(16, 12))
object_counts = clean_df['objectName'].value_counts().head(12)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA']
wedges, texts = ax.pie(object_counts.values, colors=colors[:len(object_counts)], startangle=90, shadow=True, explode=[0.05]*len(object_counts), wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
ax.set_title('Most Common Object Types in Clean Dataset', **font_title, pad=30)
legend_labels = [f"{name}: {count:,} ({count/clean_samples*100:.1f}%)" for name, count in object_counts.items()]
ax.legend(wedges, legend_labels, title="Object Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), **font_legend)
plt.tight_layout()
plt.savefig(f"{output_dir}/12_object_names_distribution_final.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 12_object_names_distribution_final.png")

print(f"\n🎉 ALL {len(os.listdir(output_dir))} PLOTS REGENERATED SUCCESSFULLY!")
print(f"✨ Directory: {output_dir}/")