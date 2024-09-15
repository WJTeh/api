from fastapi import FastAPI, File, UploadFile, Form
from transformers import ViTForImageClassification, ViTImageProcessor
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition
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

    similarity_scores = []
    image = files[0]
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    inputs = processor(images=pil_image, return_tensors="pt")
    outputs = model(**inputs)
    scene_embeddings = outputs.logits.squeeze().tolist()

    # Query Qdrant for images with the same name
# Query Qdrant for images with the same name
    results = qclient.search(
        collection_name="test",
        query_vector=scene_embeddings,
        limit=1,
        query_filter=Filter(
            must=[FieldCondition(key="name", match={"value": name})]
        )
    )

    if results and results[0].score:
        similarity_scores.append(results[0].score)

    if similarity_scores:
        return {"message": "Duplicated belonging's name found."}

 
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
        if result and "score" in result:
            similarity_scores.append(result["score"])
        else:
            print(f"Error: result missing 'score'. Result: {result}")

    if not similarity_scores:
        return {"message": "No matching images found."}

    # Compute the average similarity score
    average_similarity = mean(similarity_scores)
    print(f"Average similarity score: {average_similarity}")

    # Set a threshold to determine if the objects are similar
    threshold = 0.7

    # Determine if the objects in the query images are a match
    is_match = average_similarity >= threshold

    return {"result": {"average_similarity": average_similarity, "is_match": is_match}}
