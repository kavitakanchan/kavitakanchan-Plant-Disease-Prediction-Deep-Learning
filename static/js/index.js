document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const buttonText = document.getElementById('button-text');
    const loadingSpinner = document.getElementById('loading-spinner');

    // 1. Show Image Preview
    imageUpload.addEventListener('change', () => {
        const file = imageUpload.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreviewContainer.classList.remove('hidden');
                resultContainer.classList.add('hidden'); // Hide old results
            };
            reader.readAsDataURL(file);
        }
    });

    // 2. Handle Form Submission (AJAX)
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // Stop the page from reloading

        const file = imageUpload.files[0];
        if (!file) {
            alert('Please upload an image first.');
            return;
        }

        // Show loading state
        buttonText.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');
        resultContainer.classList.add('hidden');

        // Get the CSRF token from the form
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;

        // Create form data to send
        const formData = new FormData();
        formData.append('image', file);

        try {
            // This URL '/predict/' is defined in 'predictor/urls.py'
            const response = await fetch('/predict/', {
                method: 'POST',
                headers: {
                    // This token is required by Django for security
                    'X-CSRFToken': csrfToken,
                },
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                
                // Display the result
                resultText.innerHTML = `
                    <p class="font-bold text-2xl">${data.prediction}</p>
                    <p class="text-lg text-gray-700 mt-1">Confidence: ${data.confidence}</p>
                `;
                resultContainer.classList.remove('hidden');
            } else {
                // Handle server errors
                resultText.innerHTML = `<p class="text-red-600">Error: Could not get a prediction.</p>`;
                resultContainer.classList.remove('hidden');
            }
        } catch (error) {
            // Handle network errors
            console.error('Fetch Error:', error);
            resultText.innerHTML = `<p class="text-red-600">Error: Network problem. Please try again.</p>`;
            resultContainer.classList.remove('hidden');
        } finally {
            // Hide loading state
            buttonText.classList.remove('hidden');
            loadingSpinner.classList.add('hidden');
        }
    });
});
