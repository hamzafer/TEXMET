

import json

def find_textiles_with_thread_count():
    results = []
    with open('/home/user1/Desktop/HAMZA/THESIS/TEXMET/clean_dataset/clean_textiles_dataset.json', 'r') as f:
        data = json.load(f)

    for item in data:
        department = item.get('department', '')
        medium = item.get('medium', '')
        dimensions = item.get('dimensions', '')

        if 'warp' in medium.lower() or 'weft' in medium.lower() or 'warp' in dimensions.lower() or 'weft' in dimensions.lower():
            results.append({
                'objectID': item.get('objectID'),
                'medium': medium,
                'dimensions': dimensions,
                'department': department
            })

    with open('thread_count_results.txt', 'w') as f:
        if not results:
            f.write("No objects found with 'warp' or 'weft' in their metadata.\n")
            return

        f.write(f"Found {len(results)} objects with potential thread count information:\n\n")
        for res in results:
            f.write(f"ObjectID: {res['objectID']}\n")
            f.write(f"  Department: {res['department']}\n")
            f.write(f"  Medium: {res['medium']}\n")
            f.write(f"  Dimensions: {res['dimensions']}\n")
            f.write("-" * 20 + "\n")

if __name__ == '__main__':
    find_textiles_with_thread_count()

