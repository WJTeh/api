from fastapi import FastAPI, File, HTTPException, UploadFile, Form, Query
from transformers import ViTForImageClassification, ViTImageProcessor
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, FilterSelector
from qdrant_client.http import models

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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
from qdrant_client import models
import uuid
import io
from PIL import Image
import traceback

app = FastAPI()

# Custom response model
class UploadResponse(BaseModel):
    status: str
    message: str
    ids: List[str] = None  # Optional for when upload is successful
    name: str = None       # Optional, only include in successful upload


@app.post("/upload", response_model=UploadResponse)
async def upload_images(name: str = Form(...), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail={"status": "failed", "message": "At least one image is required."})

    try:
        # Log the incoming request for better debugging
        print(f"Received UPLOAD request with name: {name}")

        # Apply filter to check if a belonging with the same name exists in the Qdrant database
        filter_conditions = models.Filter(
            must=[
                models.FieldCondition(
                    key="name",
                    match=models.MatchValue(value=name)
                )
            ]
        )

        # Use scroll API to check if the belonging already exists
        scroll_response = qclient.scroll(
            collection_name="test",  # Replace with your collection name
            scroll_filter=filter_conditions,
            limit=1,  # We only need to check if at least one point exists
            with_payload=True,
            with_vectors=False,  # We only need payload for this check, not vectors
        )

        # Unpack the scroll response
        points, next_page_offset = scroll_response
        
        # Log the scroll response for debugging purposes
        print(f"Scroll API response: points={points}, next_page_offset={next_page_offset}")

        # If points are returned, a belonging with the same name already exists
        if points:
            return UploadResponse(
                status="failed",
                message=f"A belonging with the name '{name}' already exists. Please provide another name."
            )

    except Exception as e:
        # Log the error with traceback for debugging purposes
        error_message = f"Error during scroll operation: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        raise HTTPException(status_code=500, detail={"status": "failed", "message": "Internal Server Error occurred during duplicate check."})

    # If no duplicates, proceed with image upload and embedding generation
    image_ids = []

    try:
        for image in files:
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            inputs = processor(images=pil_image, return_tensors="pt")
            outputs = model(**inputs)
            embeddings = outputs.logits.squeeze().tolist()

            # Generate a UUID for the image
            image_id = str(uuid.uuid4())
            image_ids.append(image_id)

            # Store embeddings in Qdrant with the name as the payload
            qclient.upsert(collection_name="test", points=[{
                "id": image_id,  # Use UUID for the point ID
                "vector": embeddings,
                "payload": {"name": name}  # Store the name in the payload
            }])

    except Exception as e:
        error_message = f"Error during image processing/upload: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        raise HTTPException(status_code=500, detail={"status": "failed", "message": "Internal Server Error occurred while uploading images."})

    return UploadResponse(
        status="success",
        message="Images uploaded successfully.",
        ids=image_ids,
        name=name
    )



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
        # Log the incoming request
        print(f"Received DELETE request to delete belongings with name: {name}")

        # Apply filter to select points by 'name'
        filter_conditions = models.Filter(
            must=[
                models.FieldCondition(
                    key="name",
                    match=models.MatchValue(value=name)
                )
            ]
        )
        
        # Use scroll API to check if the point exists
        scroll_response = qclient.scroll(
            collection_name="test",  # Replace with your collection name
            scroll_filter=filter_conditions,
            limit=1,  # Limit to 1 point to check existence
            with_payload=True,
            with_vectors=False,
        )
        
        # Unpack the scroll response
        points, next_page_offset = scroll_response
        
        # Log the scroll response
        print(f"Scroll API response: points={points}, next_page_offset={next_page_offset}")

        if points:
            # If points are returned, the point exists, so proceed with deletion
            delete_response = qclient.delete(
                collection_name="test",  # Replace with your collection name
                points_selector=models.FilterSelector(filter=filter_conditions)
            )
            
            # Log the response from Qdrant
            print(f"Qdrant delete response: {delete_response}")

            if delete_response.status == "ok" or delete_response.status == "completed":
                return {"message": f"Belonging with name '{name}' deleted successfully."}
            else:
                print(f"Failed to delete belongings. Response status: {delete_response.status}")
                raise HTTPException(status_code=500, detail=f"Failed to delete belonging. Response status: {delete_response.status}")
        else:
            # Point does not exist
            return {"message": f"No belongings found with name '{name}'."}
    
    except Exception as e:
        # Log the error with traceback for debugging purposes
        import traceback
        error_message = f"Error occurred: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        raise HTTPException(status_code=500, detail="Internal Server Error occurred while deleting belongings.")