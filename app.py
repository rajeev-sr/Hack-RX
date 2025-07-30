from fastapi import FastAPI

collection = collection()
app = FastAPI()

# Helper function to convert ObjectId to string
def serialize_doc(doc):
    doc["_id"] = str(doc["_id"])
    return doc

@app.get("/")
def read_items():
    items=collection.find()
    results = [serialize_doc(doc) for doc in items]
    return results

@app.get("/items/{item_name}")
def read_item(item_name: str):
    query = {"name": item_name}
    items=collection.find(query)

    results = [serialize_doc(doc) for doc in items]
    return results

@app.post("/items/")
def create_item(item: dict):
    id=collection.insert_one(item).inserted_id
    return {"message": f"Item {id} created successfully"}

@app.put("/items/{item_name}")
def update_item(item_name: str, item: dict):
    query = {"name": item_name}
    collection.update_one(query, {"$set": item})
    return {"message": f"Item {item_name} updated successfully"}