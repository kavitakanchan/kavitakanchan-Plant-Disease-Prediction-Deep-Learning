from django.shortcuts import render
from django.http import JsonResponse
import json

# --- THIS IS THE NEW PART ---
# We import the real prediction function from the utils.py file
from .ml_files.utils import predict_image
# -----------------------------

# This view just renders the main HTML page.
def index(request):
    return render(request, 'index.html')


# --- THIS IS THE UPDATED FUNCTION ---
def predict_disease(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)
        
    image_file = request.FILES.get('image')
    if not image_file:
        return JsonResponse({'error': 'No image file provided'}, status=400)

    try:
        # Read the image file in-memory
        image_bytes = image_file.read()
        
        # --- THIS IS THE REAL PREDICTION ---
        # We call our model's helper function
        predicted_class, confidence = predict_image(image_bytes)
        # -------------------------------------

        if predicted_class is None:
            return JsonResponse({'error': 'Could not process image'}, status=500)

        # --- Format the name to be "pretty" ---
        # "Apple___Black_rot" -> "Apple: Black rot"
        try:
            plant, disease = predicted_class.split('___')
            disease = disease.replace('_', ' ')
            readable_prediction = f"{plant}: {disease}"
        except Exception as e:
            readable_prediction = predicted_class.replace('_', ' ') # Fallback
        # ----------------------------------------

        # Return the real prediction
        return JsonResponse({
            'prediction': readable_prediction,
            'confidence': f'{confidence * 100:.2f}%'
        })

    except Exception as e:
        print(f"Error in predict_disease view: {e}")
        return JsonResponse({'error': 'An unexpected error occurred'}, status=500)

