from fastapi import FastAPI, File, HTTPException, UploadFile, Form, Query
from transformers import ViTForImageClassification, ViTImageProcessor
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, FilterSelector

from PIL import Image
import io
import os
import uuid
from statistics import mean
from typing import List
from dotenv import load_dotenv

app = FastAPI(title="Feature Extraction", version="0.95.0")

load_dotenv()
qdrant_api_key = os.getenv('QDRANT_API_KEY')
qdrant_db_url = os.getenv('QDRANT_DB_URL')

# Initialize Qdrant client
qclient = QdrantClient(url=qdrant_db_url, api_key=qdrant_api_key)

# Load model and feature extractor
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

@app.post("/upload")
async def upload_images(name: str = Form(...), files: List[UploadFile] = File(...)):
    if not files:
        return {"error": "At least one image is required."}
    
    image_ids = []

    for image in files:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        inputs = processor(images=pil_image, return_tensors="pt")
        outputs = model(**inputs)
        embeddings = outputs.logits.squeeze().tolist()

        # Generate a UUID for the image
        image_id = str(uuid.uuid4())
        image_ids.append(image_id)

        # Store embeddings in Qdrant with the name
        qclient.upsert(collection_name="test", points=[{
            "id": image_id,  # Use UUID instead of filename
            "vector": embeddings,
            "payload": {"name": name}  # Store the name in the payload
        }])

    return {"status": "success", "ids": image_ids, "name": name}    
   

@app.post("/query")
async def query_image(files: List[UploadFile] = File(...), name: str = Form(...)):
    if not files:
        return {"error": "At least one image is required."}

    similarity_scores = []
    
    for image in files:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        inputs = processor(images=pil_image, return_tensors="pt")
        outputs = model(**inputs)
        scene_embeddings = outputs.logits.squeeze().tolist()

        # Query Qdrant for images with the same name
        results = qclient.search(
            collection_name="test",
            query_vector=scene_embeddings,
            limit=4,  # Adjust the limit as needed
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="name",
                        match={"value": name}
                    )
                ]
            )
        )

        # Calculate similarity scores for all matches
        for result in results:
            similarity_scores.append(result.score)

    if similarity_scores:
        # Compute the average similarity score
        average_similarity = mean(similarity_scores)
        print(f"Average similarity score: {average_similarity}")

        # Set a threshold to determine if the objects are similar
        threshold = 0.7  # You can adjust this threshold as needed

        # Determine if the objects in the query images are a match
        is_match = average_similarity >= threshold

        return {"result": {"average_similarity": average_similarity, "is_match": is_match}}
    else:
        return {"message": "No matching images found."}


@app.delete("/delete")
async def delete_belonging(name: str = Query(...)):
    try:
        # Apply filter to select points by 'name'
        filter_conditions = Filter(
            must=[
                FieldCondition(
                    key="name",
                    match=MatchValue(value=name)
                )
            ]
        )

        # Delete the points from the collection
        response = qclient.delete(
            collection_name="test",  # Replace with your collection name
            points_selector=FilterSelector(filter=filter_conditions)
        )

        # Check response or deletion count
        if response.status == "ok":
            return {"message": f"Belongings with name '{name}' deleted successfully."}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete belongings.")
    
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error occurred while deleting belongings.")