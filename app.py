from fastapi import FastAPI


app = FastAPI()

# Helper function to convert ObjectId to string
@app.get("/")
def server():
    return {"message":"server is running"}
