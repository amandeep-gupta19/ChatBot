# ChatBot


# step 1: Install all Dependencies

pip install -r requirements.txt

or

pip3 install -r requirements.txt

# step 2: Extract Data

python3 extraction.py

# step 3: Create Embeddings

python3 create_embeddings.py

# step 4: Run the Flask API

python3 app.py

# step 5: Test the API

Use Postman or curl to send POST requests to http://127.0.0.1:5000/search

Like:

curl -X POST http://127.0.0.1:5000/search -H "Content-Type: application/json" -d '{"query": "cloud computing"}'
