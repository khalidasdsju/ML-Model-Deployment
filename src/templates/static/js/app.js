// Global variables
let predictionHistory = [];
let apiBaseUrl = '';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Set API base URL
    apiBaseUrl = localStorage.getItem('apiBaseUrl') || window.location.origin;

    // Initialize navigation
    initNavigation();

    // Initialize forms
    initForms();

    // Load model info
    loadModelInfo();

    // Load prediction history from localStorage
    loadPredictionHistory();

    // Initialize charts
    initCharts();
});

// Initialize navigation
function initNavigation() {
    // Handle navigation clicks
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();

            // Get the page to show
            const pageId = this.getAttribute('data-page');

            // Update active link
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            this.classList.add('active');

            // Update page title
            document.getElementById('page-title').textContent = this.textContent.trim();

            // Hide all pages
            document.querySelectorAll('.page-content').forEach(page => page.classList.remove('active'));

            // Show the selected page
            document.getElementById(`${pageId}-page`).classList.add('active');
        });
    });

    // Toggle sidebar on mobile
    document.getElementById('toggle-sidebar').addEventListener('click', function() {
        document.getElementById('sidebar').classList.toggle('active');
    });
}

// Initialize forms
function initForms() {
    // Prediction form
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            submitPrediction();
        });
    }

    // Batch form
    const batchForm = document.getElementById('batch-form');
    if (batchForm) {
        batchForm.addEventListener('submit', function(e) {
            e.preventDefault();
            submitBatchPrediction();
        });
    }

    // Download template button
    const downloadTemplateBtn = document.getElementById('download-template');
    if (downloadTemplateBtn) {
        downloadTemplateBtn.addEventListener('click', function() {
            downloadTemplate();
        });
    }

    // Clear history button
    const clearHistoryBtn = document.getElementById('clear-history');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', function() {
            clearPredictionHistory();
        });
    }
}

// Load model info
async function loadModelInfo() {
    try {
        const response = await fetch(`${apiBaseUrl}/api/model-info`);
        const data = await response.json();

        // Update model version
        document.getElementById('model-version').textContent = data.model_version || 'Local';

        // Update important features
        if (data.important_features && data.important_features.length > 0) {
            displayImportantFeatures(data.important_features);
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Display important features
function displayImportantFeatures(features) {
    const container = document.getElementById('important-features');
    if (!container) return;

    // Clear container
    container.innerHTML = '';

    // Create feature list
    const featureList = document.createElement('ul');
    featureList.className = 'list-group';

    // Add features
    features.forEach(feature => {
        const item = document.createElement('li');
        item.className = 'list-group-item d-flex justify-content-between align-items-center';
        item.textContent = feature;
        featureList.appendChild(item);
    });

    // Add to container
    container.appendChild(featureList);
}

// Submit prediction
async function submitPrediction() {
    // Show loading state
    const resultContainer = document.getElementById('prediction-result');
    resultContainer.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Processing prediction...</p>
        </div>
    `;

    // Get form data
    const form = document.getElementById('prediction-form');
    const formData = new FormData(form);

    // Convert to JSON object
    const patientData = {};
    for (const [key, value] of formData.entries()) {
        if (value !== '') {
            patientData[key] = isNaN(value) ? value : Number(value);
        }
    }

    try {
        // Send prediction request
        const response = await fetch(`${apiBaseUrl}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(patientData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        // Display prediction result
        displayPredictionResult(result);

        // Add to history
        addToPredictionHistory(result);

    } catch (error) {
        console.error('Error making prediction:', error);
        resultContainer.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <h4 class="alert-heading">Error!</h4>
                <p>Failed to make prediction. Please try again.</p>
                <hr>
                <p class="mb-0">Error details: ${error.message}</p>
            </div>
        `;
    }
}

// Display prediction result
function displayPredictionResult(result) {
    const resultContainer = document.getElementById('prediction-result');

    // Determine result class
    const resultClass = result.prediction === 1 ? 'positive' : 'negative';
    const resultText = result.prediction === 1 ? 'High Risk' : 'Low Risk';
    const resultIcon = result.prediction === 1 ? 'bi-heart-pulse' : 'bi-heart';
    const resultColor = result.prediction === 1 ? 'danger' : 'success';

    // Format probability as percentage
    const probability = (result.probability * 100).toFixed(2);

    // Create result HTML
    resultContainer.innerHTML = `
        <div class="prediction-card ${resultClass} fade-in">
            <i class="bi ${resultIcon} display-1 text-${resultColor}"></i>
            <div class="prediction-label text-${resultColor}">${resultText}</div>
            <div class="prediction-probability">
                <span class="text-${resultColor}">${probability}%</span> probability
            </div>
            <div class="mt-3">
                <div class="progress">
                    <div class="progress-bar bg-${resultColor}" role="progressbar" style="width: ${probability}%"
                        aria-valuenow="${probability}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
            </div>
            <div class="mt-3 small text-muted">
                Threshold: ${(result.threshold * 100).toFixed(2)}%
            </div>
        </div>
    `;

    // Update feature importance if available
    if (result.features_used && result.features_used.length > 0) {
        displayFeatureImportance(result.features_used.slice(0, 10));
    }
}

// Display feature importance
function displayFeatureImportance(features) {
    const container = document.getElementById('feature-importance');
    if (!container) return;

    // Clear container
    container.innerHTML = '';

    // Create feature bars
    features.forEach((feature, index) => {
        const importance = 100 - (index * 10); // Simple decreasing importance

        const featureDiv = document.createElement('div');
        featureDiv.className = 'mb-3';

        const labelDiv = document.createElement('div');
        labelDiv.className = 'feature-label';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'feature-name';
        nameSpan.textContent = feature;

        const valueSpan = document.createElement('span');
        valueSpan.textContent = `${importance}%`;

        labelDiv.appendChild(nameSpan);
        labelDiv.appendChild(valueSpan);

        const barDiv = document.createElement('div');
        barDiv.className = 'feature-bar';
        barDiv.style.width = `${importance}%`;

        featureDiv.appendChild(labelDiv);
        featureDiv.appendChild(barDiv);

        container.appendChild(featureDiv);
    });
}

// Submit batch prediction
function submitBatchPrediction() {
    // Get file input
    const fileInput = document.getElementById('csv-file');

    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select a CSV file');
        return;
    }

    const file = fileInput.files[0];

    // Show loading state
    const resultsContainer = document.getElementById('batch-results');
    resultsContainer.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Processing batch prediction...</p>
        </div>
    `;

    // Create form data
    const formData = new FormData();
    formData.append('file', file);

    // Send batch prediction request
    fetch(`${apiBaseUrl}/api/batch-predict`, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(result => {
        displayBatchResults(result);

        // Enable download button
        document.getElementById('download-results').disabled = false;
    })
    .catch(error => {
        console.error('Error making batch prediction:', error);
        resultsContainer.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <h4 class="alert-heading">Error!</h4>
                <p>Failed to process batch prediction. Please try again.</p>
                <hr>
                <p class="mb-0">Error details: ${error.message}</p>
            </div>
        `;
    });
}

// Display batch results
function displayBatchResults(results) {
    const container = document.getElementById('batch-results');

    // Create summary card
    const summaryHtml = `
        <div class="row mb-4">
            <div class="col-md-4">
                <div class="card bg-primary text-white">
                    <div class="card-body text-center">
                        <h5 class="card-title">Total Predictions</h5>
                        <h2>${results.summary.total_patients}</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-success text-white">
                    <div class="card-body text-center">
                        <h5 class="card-title">Negative Results</h5>
                        <h2>${results.summary.negative_predictions}</h2>
                        <p>${results.summary.negative_percentage.toFixed(1)}%</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-danger text-white">
                    <div class="card-body text-center">
                        <h5 class="card-title">Positive Results</h5>
                        <h2>${results.summary.positive_predictions}</h2>
                        <p>${results.summary.positive_percentage.toFixed(1)}%</p>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Create results table
    let tableHtml = `
        <div class="table-responsive">
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Probability</th>
                        <th>Prediction</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
    `;

    // Add rows
    results.predictions.forEach((pred, index) => {
        const resultClass = pred.prediction === 1 ? 'danger' : 'success';
        const resultText = pred.prediction === 1 ? 'High Risk' : 'Low Risk';

        tableHtml += `
            <tr>
                <td>${index + 1}</td>
                <td>${(pred.probability * 100).toFixed(2)}%</td>
                <td><span class="badge bg-${resultClass}">${resultText}</span></td>
                <td>
                    <button class="btn btn-sm btn-outline-primary view-details" data-index="${index}">
                        View Details
                    </button>
                </td>
            </tr>
        `;
    });

    tableHtml += `
                </tbody>
            </table>
        </div>
    `;

    // Set HTML
    container.innerHTML = summaryHtml + tableHtml;

    // Add event listeners to view details buttons
    document.querySelectorAll('.view-details').forEach(button => {
        button.addEventListener('click', function() {
            const index = parseInt(this.getAttribute('data-index'));
            showPredictionDetails(results.predictions[index]);
        });
    });

    // Store results for download
    window.batchResults = results;
}

// Show prediction details in modal
function showPredictionDetails(prediction) {
    const modal = new bootstrap.Modal(document.getElementById('prediction-details-modal'));
    const modalContent = document.getElementById('prediction-details-content');

    // Format patient data
    let patientDataHtml = '<h6>Patient Data</h6><div class="row">';

    for (const [key, value] of Object.entries(prediction.patient_data)) {
        patientDataHtml += `
            <div class="col-md-4 mb-2">
                <strong>${key}:</strong> ${value}
            </div>
        `;
    }

    patientDataHtml += '</div>';

    // Format prediction result
    const resultClass = prediction.prediction === 1 ? 'danger' : 'success';
    const resultText = prediction.prediction === 1 ? 'High Risk' : 'Low Risk';

    const resultHtml = `
        <div class="alert alert-${resultClass} mb-4">
            <h4 class="alert-heading">${resultText}</h4>
            <p>Probability: ${(prediction.probability * 100).toFixed(2)}%</p>
            <p>Threshold: ${(prediction.threshold * 100).toFixed(2)}%</p>
            <p>Model Version: ${prediction.model_version}</p>
            <p>Timestamp: ${prediction.timestamp}</p>
        </div>
    `;

    // Set modal content
    modalContent.innerHTML = resultHtml + patientDataHtml;

    // Show modal
    modal.show();
}

// Download template CSV
function downloadTemplate() {
    // Create CSV content
    const requiredFields = [
        'FS', 'DT', 'NYHA', 'HR', 'BNP', 'LVIDs', 'BMI', 'LAV',
        'Wall_Subendocardial', 'LDLc', 'Age', 'ECG_T_inversion',
        'ICT', 'RBS', 'EA', 'Chest_pain'
    ];

    const optionalFields = [
        'LVEF', 'Sex', 'HTN', 'DM', 'Smoker', 'DL', 'TropI', 'RWMA', 'MR'
    ];

    const header = [...requiredFields, ...optionalFields].join(',');
    const sampleRow = [
        '25', '160', '3', '95', '800', '4.8', '28.5', '45',
        '1', '140', '65', '1', '110', '180', '0.8', '1',
        '45', '1', '1', '1', '1', '1', '0.5', '1', '1'
    ].join(',');

    const csvContent = `${header}\n${sampleRow}`;

    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'heart_failure_prediction_template.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Download batch results
document.getElementById('download-results').addEventListener('click', function() {
    if (!window.batchResults) return;

    // Create CSV content
    let csvContent = 'Index,Probability,Prediction\n';

    window.batchResults.predictions.forEach((pred, index) => {
        const resultText = pred.prediction === 1 ? 'High Risk' : 'Low Risk';
        csvContent += `${index + 1},${(pred.probability * 100).toFixed(2)}%,${resultText}\n`;
    });

    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'batch_prediction_results.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

// Load prediction history
function loadPredictionHistory() {
    const historyJson = localStorage.getItem('predictionHistory');
    if (historyJson) {
        predictionHistory = JSON.parse(historyJson);
        updateHistoryDisplay();
        updateDashboardStats();
    }
}

// Add to prediction history
function addToPredictionHistory(prediction) {
    // Add patient ID if not present
    if (!prediction.patientId) {
        prediction.patientId = `P${Date.now().toString().slice(-6)}`;
    }

    // Add to history
    predictionHistory.unshift(prediction);

    // Limit history size
    if (predictionHistory.length > 100) {
        predictionHistory = predictionHistory.slice(0, 100);
    }

    // Save to localStorage
    localStorage.setItem('predictionHistory', JSON.stringify(predictionHistory));

    // Update displays
    updateHistoryDisplay();
    updateDashboardStats();
}

// Update history display
function updateHistoryDisplay() {
    const historyList = document.getElementById('history-list');
    const recentPredictions = document.getElementById('recent-predictions');

    if (!historyList || !recentPredictions) return;

    // Create HTML for history items
    let historyHtml = '';
    let recentHtml = '';

    if (predictionHistory.length === 0) {
        historyHtml = '<tr><td colspan="5" class="text-center">No prediction history</td></tr>';
        recentHtml = '<tr><td colspan="5" class="text-center">No recent predictions</td></tr>';
    } else {
        predictionHistory.forEach((pred, index) => {
            const resultClass = pred.prediction === 1 ? 'danger' : 'success';
            const resultText = pred.prediction === 1 ? 'High Risk' : 'Low Risk';
            const row = `
                <tr>
                    <td>${pred.timestamp}</td>
                    <td>${pred.patientId}</td>
                    <td>${(pred.probability * 100).toFixed(2)}%</td>
                    <td><span class="badge bg-${resultClass}">${resultText}</span></td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary view-history-details" data-index="${index}">
                            View
                        </button>
                    </td>
                </tr>
            `;

            historyHtml += row;

            // Only add to recent predictions if in the first 5
            if (index < 5) {
                recentHtml += row;
            }
        });
    }

    // Update DOM
    historyList.innerHTML = historyHtml;
    recentPredictions.innerHTML = recentHtml;

    // Add event listeners
    document.querySelectorAll('.view-history-details').forEach(button => {
        button.addEventListener('click', function() {
            const index = parseInt(this.getAttribute('data-index'));
            showPredictionDetails(predictionHistory[index]);
        });
    });
}

// Clear prediction history
function clearPredictionHistory() {
    if (confirm('Are you sure you want to clear all prediction history?')) {
        predictionHistory = [];
        localStorage.removeItem('predictionHistory');
        updateHistoryDisplay();
        updateDashboardStats();
    }
}

// Update dashboard statistics
function updateDashboardStats() {
    const totalPredictions = document.getElementById('total-predictions');
    const positivePredictions = document.getElementById('positive-predictions');
    const negativePredictions = document.getElementById('negative-predictions');

    if (!totalPredictions || !positivePredictions || !negativePredictions) return;

    // Calculate stats
    const total = predictionHistory.length;
    const positive = predictionHistory.filter(p => p.prediction === 1).length;
    const negative = total - positive;

    // Update DOM
    totalPredictions.textContent = total;
    positivePredictions.textContent = positive;
    negativePredictions.textContent = negative;

    // Update chart
    updatePredictionChart(positive, negative);
}

// Initialize charts
function initCharts() {
    // Create prediction distribution chart
    const ctx = document.getElementById('prediction-chart');
    if (!ctx) return;

    window.predictionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['High Risk', 'Low Risk'],
            datasets: [{
                data: [0, 0],
                backgroundColor: ['#e74a3b', '#1cc88a'],
                hoverBackgroundColor: ['#e02d1b', '#13a673'],
                hoverBorderColor: 'rgba(234, 236, 244, 1)',
            }]
        },
        options: {
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            },
            cutout: '70%'
        }
    });

    // Update with initial data
    const positive = predictionHistory.filter(p => p.prediction === 1).length;
    const negative = predictionHistory.length - positive;
    updatePredictionChart(positive, negative);
}

// Update prediction chart
function updatePredictionChart(positive, negative) {
    if (!window.predictionChart) return;

    window.predictionChart.data.datasets[0].data = [positive, negative];
    window.predictionChart.update();
}
