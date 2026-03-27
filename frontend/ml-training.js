// API Configuration
const API_URL = 'http://localhost:8001';
let cleanedDataStats = null;

document.addEventListener('DOMContentLoaded', () => {
    initializeMLTrainingPage();
});

function initializeMLTrainingPage() {
    // Get cleaned data stats from sessionStorage
    const statsJson = sessionStorage.getItem('cleanedDataStats');

    if (!statsJson) {
        // For now, don't redirect - just show error
        showError('No cleaned data available. Please complete the cleaning process first.');
        return;
    }

    cleanedDataStats = JSON.parse(statsJson);

    // Set up event listeners
    const backBtn = document.getElementById('backBtn');
    const trainBtn = document.getElementById('trainBtn');
    const downloadModelBtn = document.getElementById('downloadModelBtn');
    const suggestionsBtn = document.getElementById('suggestionsBtn');

    if (backBtn) {
        backBtn.addEventListener('click', () => {
            window.location.href = 'cleaning-results.html';
        });
    }

    if (trainBtn) {
        trainBtn.addEventListener('click', trainMLModels);
    }

    if (downloadModelBtn) {
        downloadModelBtn.addEventListener('click', downloadBestModel);
    }

    if (suggestionsBtn) {
        suggestionsBtn.addEventListener('click', () => {
            window.location.href = 'suggestions.html';
        });
    }

    // Populate target column dropdown
    populateTargetColumns();
}

function populateTargetColumns() {
    const targetSelect = document.getElementById('targetColumn');
    const dataStatus = document.getElementById('dataStatus');

    if (!cleanedDataStats || !cleanedDataStats.columns) {
        showError('No dataset information available');
        if (dataStatus) dataStatus.style.display = 'none';
        return;
    }

    // Clear existing options except the first one
    while (targetSelect.options.length > 1) {
        targetSelect.remove(1);
    }

    // Add all columns as options
    cleanedDataStats.columns.forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.textContent = `${col} (${cleanedDataStats.data_types[col] || 'unknown'})`;
        targetSelect.appendChild(option);
    });

    // Show success message
    if (cleanedDataStats.columns.length > 0) {
        console.log(`ML Training: Successfully populated ${cleanedDataStats.columns.length} columns`);
        if (dataStatus) dataStatus.style.display = 'block';
    } else {
        showError('No columns available for training');
        if (dataStatus) dataStatus.style.display = 'none';
    }
}

async function trainMLModels() {
    const targetColumn = document.getElementById('targetColumn').value;

    if (!targetColumn) {
        showError('Please select a target column');
        return;
    }

    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('errorMessage').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';

    try {
        const response = await fetch(`${API_URL}/train-ml`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                target_column: targetColumn
            })
        });

        const result = await response.json();

        if (result.success) {
            displayMLResults(result);
        } else {
            showError(result.error || 'Error training ML models');
        }
    } catch (error) {
        console.error('ML training error:', error);
        showError('Error connecting to backend for ML training');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function displayMLResults(result) {
    const resultsSection = document.getElementById('resultsSection');

    // Fill summary info
    document.getElementById('problemType').textContent = result.problem_type;
    document.getElementById('targetCol').textContent = result.target_column;
    document.getElementById('totalSamples').textContent = result.dataset_info.total_samples.toLocaleString();
    document.getElementById('numFeatures').textContent = result.dataset_info.features;
    document.getElementById('trainTestSplit').textContent = `${result.dataset_info.train_samples}/${result.dataset_info.test_samples}`;

    // Display best model
    document.getElementById('bestModelName').textContent = result.best_model;

    // Show results
    displayModelComparison(result.results, result.problem_type, result.best_model);

    resultsSection.style.display = 'block';

    // Store results for later use
    sessionStorage.setItem('mlResults', JSON.stringify(result));
}

function displayModelComparison(results, problemType, bestModel) {
    const container = document.getElementById('modelsComparison');
    container.innerHTML = '';

    const isClassification = problemType === 'classification';

    // Create table
    const table = document.createElement('table');
    table.className = 'results-table';

    // Table header
    const thead = document.createElement('thead');
    let headerHTML = '<tr><th>Model</th>';

    if (isClassification) {
        headerHTML += '<th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>CV Mean</th>';
    } else {
        headerHTML += '<th>MSE</th><th>RMSE</th><th>R² Score</th><th>CV RMSE</th>';
    }

    headerHTML += '<th>Status</th></tr>';
    thead.innerHTML = headerHTML;
    table.appendChild(thead);

    // Table body
    const tbody = document.createElement('tbody');

    Object.entries(results).forEach(([modelName, metrics]) => {
        const row = document.createElement('tr');
        if (modelName === bestModel) {
            row.className = 'best-model-row';
        }

        let rowHTML = `<td><strong>${modelName}</strong>${modelName === bestModel ? ' 🏆' : ''}</td>`;

        if ('error' in metrics) {
            rowHTML += `<td colspan="${isClassification ? 5 : 4}" class="error-cell">❌ ${metrics.error}</td><td>Failed</td>`;
        } else {
            if (isClassification) {
                rowHTML += `
                    <td>${metrics.accuracy.toFixed(4)}</td>
                    <td>${metrics.precision.toFixed(4)}</td>
                    <td>${metrics.recall.toFixed(4)}</td>
                    <td>${metrics.f1_score.toFixed(4)}</td>
                    <td>${metrics.cv_mean.toFixed(4)} ± ${metrics.cv_std.toFixed(4)}</td>
                    <td>✅ Trained</td>
                `;
            } else {
                rowHTML += `
                    <td>${metrics.mse.toFixed(4)}</td>
                    <td>${metrics.rmse.toFixed(4)}</td>
                    <td>${metrics.r2_score.toFixed(4)}</td>
                    <td>${metrics.cv_rmse.toFixed(4)} ± ${metrics.cv_std.toFixed(4)}</td>
                    <td>✅ Trained</td>
                `;
            }
        }

        row.innerHTML = rowHTML;
        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    container.appendChild(table);

    // Update best model score display
    if (bestModel && results[bestModel] && !('error' in results[bestModel])) {
        const bestMetrics = results[bestModel];
        let scoreText = '';

        if (isClassification) {
            scoreText = `F1-Score: ${bestMetrics.f1_score.toFixed(4)} | Accuracy: ${bestMetrics.accuracy.toFixed(4)}`;
        } else {
            scoreText = `RMSE: ${bestMetrics.rmse.toFixed(4)} | R²: ${bestMetrics.r2_score.toFixed(4)}`;
        }

        document.getElementById('bestModelScore').textContent = scoreText;
    }
}

async function downloadBestModel() {
    const mlResults = JSON.parse(sessionStorage.getItem('mlResults'));
    if (!mlResults) {
        showError('No ML results available. Please train models first.');
        return;
    }

    try {
        const response = await fetch(`${API_URL}/download-best-model`);

        if (!response.ok) {
            let errorMessage = 'Unable to download best model.';
            try {
                const errorData = await response.json();
                if (errorData && errorData.error) {
                    errorMessage = errorData.error;
                }
            } catch (parseError) {
                // Keep fallback message if response is not JSON.
            }
            showError(errorMessage);
            return;
        }

        const blob = await response.blob();
        const contentDisposition = response.headers.get('Content-Disposition') || '';
        let filename = 'best_model.pkl';

        const filenameMatch = contentDisposition.match(/filename=\"?([^\";]+)\"?/i);
        if (filenameMatch && filenameMatch[1]) {
            filename = filenameMatch[1];
        }

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        showSuccess(`Download started: ${filename}`);
    } catch (error) {
        console.error('Download model error:', error);
        showError('Error connecting to backend while downloading best model.');
    }
}

function showSuccess(message) {
    const successDiv = document.getElementById('successMessage');
    const errorDiv = document.getElementById('errorMessage');

    if (errorDiv) {
        errorDiv.style.display = 'none';
    }

    if (successDiv) {
        successDiv.textContent = message;
        successDiv.style.display = 'block';

        setTimeout(() => {
            successDiv.style.display = 'none';
        }, 3500);
    }
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    const successDiv = document.getElementById('successMessage');

    if (successDiv) {
        successDiv.style.display = 'none';
    }

    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    }

    // Hide success message if showing error
    const dataStatus = document.getElementById('dataStatus');
    if (dataStatus) {
        dataStatus.style.display = 'none';
    }
}