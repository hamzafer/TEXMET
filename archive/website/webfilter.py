import json
import os
from pathlib import Path

def create_html_gallery():
    """Create simple HTML gallery for quick review"""
    
    # Load your dataset
    with open("/home/user1/Desktop/HAMZA/THESIS/TEXMET/data/FINAL_CORRECTED_MET_TEXTILES_DATASET/objects_with_images_only/ALL_TEXTILES_AND_TAPESTRIES_WITH_IMAGES_20250705_230315.json", "r") as f:
        data = json.load(f)
    
    images_dir = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/download/MET_TEXTILES_BULLETPROOF_DATASET/images"
    
    # Group by department for easier review
    by_dept = {}
    total_with_images = 0
    
    for obj in data:
        dept = obj.get('department', 'Unknown')
        if dept not in by_dept:
            by_dept[dept] = []
        
        # Check if image exists
        obj_id = str(obj['objectID'])
        img_files = [f for f in os.listdir(images_dir) if f.startswith(obj_id)]
        
        if img_files:
            by_dept[dept].append(obj)
            total_with_images += 1
    
    html = f"""
    <!DOCTYPE html>
    <html><head><title>MET Textiles Review - {total_with_images:,} Images</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ position: sticky; top: 0; background: white; padding: 10px 0; border-bottom: 2px solid #ccc; margin-bottom: 20px; }}
        .dept {{ margin: 20px 0; border: 1px solid #ccc; padding: 10px; }}
        .dept h2 {{ margin: 0 0 10px 0; background: #f5f5f5; padding: 10px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }}
        .item {{ border: 1px solid #ddd; padding: 5px; cursor: pointer; position: relative; }}
        .item img {{ width: 100%; height: 150px; object-fit: cover; }}
        .reject {{ background: #ffcccc; border: 2px solid #ff0000; }}
        .counter {{ position: absolute; top: 5px; right: 5px; background: rgba(0,0,0,0.7); color: white; padding: 2px 6px; border-radius: 3px; font-size: 12px; }}
        .info {{ font-size: 12px; margin-top: 5px; }}
        .controls {{ margin: 10px 0; }}
        button {{ padding: 10px 20px; margin-right: 10px; font-size: 16px; }}
        .export {{ background: #4CAF50; color: white; border: none; border-radius: 5px; }}
        .clear {{ background: #f44336; color: white; border: none; border-radius: 5px; }}
    </style>
    <script>
        let deptCounters = {{}};
        
        function initCounters() {{
            // Initialize counters for each department
            document.querySelectorAll('.dept').forEach(dept => {{
                const deptName = dept.dataset.department;
                deptCounters[deptName] = 0;
            }});
            updateAllCounters();
        }}
        
        function toggleReject(elem) {{
            const deptName = elem.closest('.dept').dataset.department;
            
            if (elem.classList.contains('reject')) {{
                elem.classList.remove('reject');
                deptCounters[deptName]--;
            }} else {{
                elem.classList.add('reject');
                deptCounters[deptName]++;
            }}
            updateAllCounters();
        }}
        
        function updateAllCounters() {{
            const totalRejects = Object.values(deptCounters).reduce((a, b) => a + b, 0);
            const totalImages = {total_with_images};
            const totalKeeps = totalImages - totalRejects;
            
            document.getElementById('totalCounts').innerHTML = 
                `Total: ${{totalImages:,}} | Rejects: ${{totalRejects:,}} | Keeps: ${{totalKeeps:,}}`;
            
            // Update department counters
            Object.keys(deptCounters).forEach(dept => {{
                const deptDiv = document.querySelector(`[data-department="${{dept}}"]`);
                const deptItems = deptDiv.querySelectorAll('.item').length;
                const deptRejects = deptCounters[dept];
                const deptKeeps = deptItems - deptRejects;
                
                deptDiv.querySelector('h2').innerHTML = 
                    `${{dept}} | Total: ${{deptItems}} | Rejects: ${{deptRejects}} | Keeps: ${{deptKeeps}}`;
            }});
            
            // Update individual counters
            document.querySelectorAll('.dept').forEach(dept => {{
                const items = dept.querySelectorAll('.item');
                items.forEach((item, index) => {{
                    const counter = item.querySelector('.counter');
                    counter.textContent = index + 1;
                }});
            }});
        }}
        
        function exportRejects() {{
            const rejects = Array.from(document.querySelectorAll('.reject')).map(el => el.dataset.objectId);
            console.log('REJECT IDs:', rejects);
            
            // Create downloadable file
            const blob = new Blob([JSON.stringify(rejects, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'textile_rejects.json';
            a.click();
            
            alert(`Exported ${{rejects.length}} reject IDs to textile_rejects.json`);
        }}
        
        function clearAll() {{
            document.querySelectorAll('.reject').forEach(el => el.classList.remove('reject'));
            Object.keys(deptCounters).forEach(key => deptCounters[key] = 0);
            updateAllCounters();
        }}
        
        window.onload = initCounters;
    </script>
    </head><body>
    <div class="header">
        <h1>MET Textiles Review - REJECT Mode</h1>
        <div class="controls">
            <span id="totalCounts">Loading...</span>
            <br><br>
            <button class="export" onclick="exportRejects()">📥 Export Reject List</button>
            <button class="clear" onclick="clearAll()">🗑️ Clear All Rejects</button>
        </div>
        <p><strong>Instructions:</strong> Click images to mark as REJECT (red). Unmarked images are KEEPS. Export when done.</p>
    </div>
    """
    
    # Sort departments by count (largest first)
    sorted_depts = sorted(by_dept.items(), key=lambda x: len(x[1]), reverse=True)
    
    for dept, objects in sorted_depts:
        html += f'<div class="dept" data-department="{dept}"><h2>{dept} ({len(objects)} items)</h2><div class="grid">'
        
        # Show ALL objects (removed the [:50] limit)
        for i, obj in enumerate(objects):
            obj_id = str(obj['objectID'])
            img_files = [f for f in os.listdir(images_dir) if f.startswith(obj_id)]
            
            if img_files:
                img_path = f"{images_dir}/{img_files[0]}"
                title = obj.get('title', '')[:40] + "..." if len(obj.get('title', '')) > 40 else obj.get('title', '')
                
                html += f'''
                <div class="item" data-object-id="{obj_id}" onclick="toggleReject(this)">
                    <div class="counter">{i+1}</div>
                    <img src="{img_path}" alt="{title}" loading="lazy">
                    <div class="info">
                        <strong>ID:</strong> {obj_id}<br>
                        <strong>Title:</strong> {title}<br>
                        <strong>Class:</strong> {obj.get('classification', '')}
                    </div>
                </div>
                '''
        
        html += '</div></div>'
    
    html += '</body></html>'
    
    with open("textile_review_gallery.html", "w") as f:
        f.write(html)
    
    print("📝 Created textile_review_gallery.html")
    print(f"📊 Total images: {total_with_images:,}")
    print("🌐 Open in browser to review images")
    print("🔴 Click images to mark as REJECT")
    print("💾 Export reject list when done")

if __name__ == "__main__":
    create_html_gallery()