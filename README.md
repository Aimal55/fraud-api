##  Mini-Gateway Fraud & OCR Prototype – 

It demonstrates a working MVP pipeline for onboarding micro-merchants by evaluating fraud risk and extracting receipt information from uploaded images.


The system consists of:
- A fraud detection model trained on synthetic but realistic transaction data
- An OCR pipeline to extract merchant name and total amount from receipt images
- A REST API (built with FastAPI) that returns fraud score + receipt fields
- Dockerized deployment for fast, reproducible testing

---

##  Project Structure
karsaazassignment/
├── app/
│ ├── init.py
│ ├── main.py # FastAPI API entrypoint
│
├── dataeda/
│ └── dataeda2.ipynb # EDA and insights
│
├── datagenerator/
│ ├── csvdatagen.py # Generate transaction CSV
│ └── ocrdatagen.py # Generate fake receipt images
│
├── dataset/
│ ├── receipts/ # Sample receipt images
│ └── transactions/ # CSV data for transactions
│
├── model/
│ ├── saved_model/ # Trained model output
│ ├── train.py # Model training script
│ └── trymodel.ipynb # Testing notebook
│
├── ocr_pipeline/
│ ├── init.py
│ └── pipeline.py # OCR logic using Tesseract
│
├── Dockerfile
├── requirements.txt
├── ocr_results.json # Output of OCR pipeline
└── README.md

## How to run the docker file 

### 1.  Build the Docker image

#bash
docker build -t fraud-ocr-api .

###  2.  Run the Docker container
# Note: The Images and transaction.csv are already included in the build

docker run -it --rm -p 8000:8000 fraud-ocr-api

docker ps 

API Endpoint – /score
POST http://localhost:8000/score

## 3.SAMPLE PAYLOAD 

{
  "transaction": {
    "amount": 389.10,
    "bin": 412456,
    "device_id": "dev_1023",
    "geo": "33.6844,73.0479"
  },
  "receipt_path": "/app/dataset/receipts/receipt_002.jpg" 
}

## 4.SAMPLE RESPONSE

{
  "fraud_score": 1,
  "merchant_name": "Book Haven",
  "total": 50.89
}

## NOTE 
You can regenerate fake data and receipts using csvdatagen.py and ocrdatagen.py
OCR results are saved in ocr_results.json during testing