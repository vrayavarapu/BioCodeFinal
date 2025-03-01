
// Load TensorFlow.js model and metadata
let model;
let labelsList;

async function loadModel() {
    const modelStatus = document.getElementById('model-status');
    const inputContainer = document.getElementById('input-container');
    
    try {
        console.log("Starting model load from /model/model.json");
        model = await tf.loadLayersModel('/model/model.json');
        
        // Load the labels
        const metadataResponse = await fetch('/model/metadata.json');
        const metadata = await metadataResponse.json();
        labelsList = metadata.labels || 
            ["Class 1", "Class 2", "Class 3", "Class 4"]; // Default labels if none found
        
        console.log("Loaded labels:", labelsList);
        console.log("Model loaded successfully:", JSON.stringify(model.toJSON()).substring(0, 1000) + "…[TRUNCATED]");
        
        // Update UI
        modelStatus.className = 'alert alert-success';
        modelStatus.innerHTML = '<i class="bi bi-check-circle me-2"></i> Model loaded successfully!';
        inputContainer.style.display = 'block';
    } catch (error) {
        console.error("Error loading model:", error);
        modelStatus.className = 'alert alert-danger';
        modelStatus.innerHTML = `
            <i class="bi bi-exclamation-triangle me-2"></i>
            Failed to load model: ${error.message}
        `;
    }
}

// Run inference on the selected image
async function runInference() {
    const previewImage = document.getElementById('preview-image');
    const resultContainer = document.getElementById('result-container');
    const inferenceResult = document.getElementById('inference-result');
    
    if (!previewImage.src || previewImage.style.display === 'none') {
        inferenceResult.innerHTML = `
            <div class="alert alert-warning">
                <i class="bi bi-exclamation-circle me-2"></i>
                Please select an image first.
            </div>`;
        resultContainer.style.display = 'block';
        return;
    }
    
    try {
        // Prepare the image
        const image = new Image();
        image.src = previewImage.src;
        
        // Create a tensor from the image
        const inputTensor = tf.browser.fromPixels(image)
            .resizeNearestNeighbor([224, 224]) // Resize to model input size
            .toFloat()
            .div(tf.scalar(255.0))  // Normalize to [0,1]
            .expandDims();          // Add batch dimension
        
        console.log("Running inference on image");
        
        // Run prediction
        const predictions = model.predict(inputTensor);
        const predictionData = await predictions.data();
        
        console.log("Inference result:", predictionData);
        
        // Process results
        let results = [];
        for (let i = 0; i < predictionData.length; i++) {
            results.push({
                class: labelsList[i],
                confidence: predictionData[i]
            });
        }
        
        // Sort by confidence (highest first)
        results.sort((a, b) => b.confidence - a.confidence);
        
        // Create results display
        let resultHTML = '<ul class="list-group mb-4">';
        results.forEach(result => {
            const backgroundColor = result.confidence > 0.5 ? 'bg-success' : 'bg-secondary';
            
            resultHTML += `
            <li class="list-group-item d-flex justify-content-between align-items-center">
                <span>${result.class}</span>
                <span class="badge ${backgroundColor} rounded-pill">
                    <i class="bi bi-check-circle"></i>
                </span>
            </li>`;
        });
        resultHTML += '</ul>';
        
        // Add custom feedback based on top classification
        const topResult = results[0].class;
        let feedbackHTML = '<div class="card bg-dark"><div class="card-body">';
        feedbackHTML += '<h5 class="card-title text-white">Treatment Recommendations</h5>';
        
        if (topResult === "T1") {
            feedbackHTML += `
                <p class="card-text text-white">T1 (Tumor ≤3 cm in greatest dimension, confined to the cerebellum):</p>
                <ul class="text-white">
                    <li>Surgical Resection: The primary treatment for a localized tumor (T1) is surgical removal of the tumor. Since it is confined to the cerebellum, the goal is to completely remove the tumor if feasible.</li>
                    <li>Post-Surgical Radiation Therapy: After surgery, radiation therapy may be used to target any remaining tumor cells and reduce the risk of recurrence.</li>
                    <li>Chemotherapy: Chemotherapy might be administered, especially for patients who are younger or in cases where the tumor is difficult to remove completely.</li>
                </ul>
            `;
        } else if (topResult === "T2") {
            feedbackHTML += `
                <p class="card-text text-white">T2 (Tumor >3 cm but still localized in the cerebellum):</p>
                <ul class="text-white">
                    <li>Surgical Resection: As with T1, surgery is the primary treatment, although it might be more complex due to the larger tumor size.</li>
                    <li>Radiation Therapy: After surgery, radiation therapy to the entire brain and spinal cord is generally administered to treat any microscopic disease that might be left behind.</li>
                    <li>Chemotherapy: Chemotherapy can be used both during and after radiation, particularly in younger patients or those with high-risk features.</li>
                </ul>
            `;
        } else if (topResult === "Not Pediatric Medulloblastoma but still bad") {
            feedbackHTML += `
                <p class="card-text text-white">A Tumor but not Medulloblastoma:</p>
                <p class="card-text text-white">No Pediatric Medulloblastoma, but we believe you have another type of tumor. Please consult with a medical professional for more information.</p>
            `;
        } else {
            feedbackHTML += `
                <p class="card-text text-white">No tumor detected. Please consult with a medical professional for a complete diagnosis.</p>
            `;
        }
        
        feedbackHTML += '<div class="alert alert-warning mt-3"><i class="bi bi-exclamation-triangle me-2"></i> This is a demo application and not a substitute for professional medical advice.</div>';
        feedbackHTML += '</div></div>';
        
        // Update UI
        resultContainer.style.display = 'block';
        inferenceResult.innerHTML = resultHTML + feedbackHTML;

        // Cleanup
        inputTensor.dispose();
        predictions.dispose();
    } catch (error) {
        console.error('Error during inference:', error);
        inferenceResult.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle me-2"></i>
                Error during inference: ${error.message}
            </div>`;
    }
}

// Set up image input preview
document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('image-input');
    const previewImage = document.getElementById('preview-image');
    
    if (imageInput) {
        imageInput.addEventListener('change', function(e) {
            if (e.target.files && e.target.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewImage.style.display = 'block';
                };
                
                reader.readAsDataURL(e.target.files[0]);
            }
        });
    }
});

// Load model when page loads
window.addEventListener('load', loadModel);
