import os
import base64
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# We need the Flask app context to query the DB
from app import app
from models import db, CatalogEntry

IMGBB_API_KEY = 'cc532d52d3ec271f34a5dd8227db219a'

def upload_to_imgbb(filepath):
    print(f"Uploading {filepath} to ImgBB...")
    with open(filepath, "rb") as f:
        image_content = f.read()

    b64_image = base64.b64encode(image_content).decode('utf-8')
    imgbb_res = requests.post(
        "https://api.imgbb.com/1/upload",
        data={
            "key": IMGBB_API_KEY,
            "image": b64_image,
            "name": os.path.basename(filepath)
        },
        timeout=30.0
    )

    if imgbb_res.status_code == 200:
        url = imgbb_res.json()['data']['url']
        return url
    else:
        print(f"Failed to upload: {imgbb_res.text}")
        return None


def main():
    with app.app_context():
        # Get all entries with an image_url
        entries = CatalogEntry.query.filter(CatalogEntry.image_url != None).all()
        print(f"Found {len(entries)} entries with an image_url.")
        
        updated_count = 0
        
        for entry in entries:
            # Check if the URL is a local or old server URL
            if 'static/uploads/' in entry.image_url and 'i.ibb.co' not in entry.image_url:
                print(f"Processing Entry ID: {entry.id}")
                print(f"Old URL: {entry.image_url}")
                
                # Extract filename
                filename = entry.image_url.split('/')[-1]
                
                # Construct local file path
                local_path = os.path.join(app.root_path, 'static', 'uploads', filename)
                
                if os.path.exists(local_path):
                    # Upload to ImgBB
                    new_url = upload_to_imgbb(local_path)
                    if new_url:
                        print(f"New URL: {new_url}")
                        entry.image_url = new_url
                        updated_count += 1
                        # Commit incrementally to be safe
                        db.session.commit()
                else:
                    print(f"Local file not found for {filename}, skipping.")

        print(f"Finished. Updated {updated_count} entries.")

if __name__ == "__main__":
    main()
