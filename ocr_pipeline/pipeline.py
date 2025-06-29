import cv2
import pytesseract
import re
import difflib
import json
import os

merchant_candidates = [
    "Cafe Luna", "Quick Mart", "Book Haven", "Tech Store",
    "Green Grocery", "Urban Outfit", "Fresh Bites", "Gear Hub"
]

def closest_merchant(lines):
    best_match, best_score = "UNKNOWN", 0
    for line in lines:
        matches = difflib.get_close_matches(line, merchant_candidates, n=1, cutoff=0.3)
        if matches:
            score = difflib.SequenceMatcher(None, line, matches[0]).ratio()
            if score > best_score:
                best_match, best_score = matches[0], score
    return best_match

def run_ocr(receipt_path):
    """
    Process a single receipt image at receipt_path.
    Returns {"merchant_name": str, "total": float or 'NOT_FOUND'}.
    """
    filename = os.path.basename(receipt_path)

    # Load image
    img = cv2.imread(receipt_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f" Could not read {filename}")
        return {"merchant_name": "READ_ERROR", "total": "NOT_FOUND"}

    # adaptive contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        img_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    # Crop to largest contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = img_clahe
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        roi = img_clahe[y:y + h, x:x + w]

    # OCR
    text = pytesseract.image_to_string(roi, config="--oem 1 --psm 6")

    # Process lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Merchant matching
    merchant_name = closest_merchant(lines)

    # Amount parsing
    total = None
    amount_pattern = re.compile(r'(?i)(total|amount|amt)[^\d]*([\d,.]+)')
    for line in lines:
        match = amount_pattern.search(line)
        if match:
            try:
                total = float(match.group(2).replace(",", "").strip())
            except ValueError:
                total = None
            break

    print(f" Processed {filename}: {merchant_name} | Total: {total}")

    return {
        "merchant_name": merchant_name,
        "total": total if total is not None else "NOT_FOUND"
    }

# main function of pipeline 
if __name__ == "__main__":
    from glob import glob
    results = {}
    for path in glob("dataset/receipts/*.jpg"):
        results[os.path.basename(path)] = run_ocr(path)
    with open("ocr_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("OCR results saved to ocr_results.json")