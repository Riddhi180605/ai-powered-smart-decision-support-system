// API Configuration - Dynamic for both development and production
function getAPIUrl() {
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return 'http://localhost:8001';
    }
    return window.location.origin;
}

const API_URL = getAPIUrl();
let originalData = null;
let datasetStats = null;
let cleaningConfig = {
    removeDuplicates: true,
    deleteHighMissing: true,
    deleteLowMissing: true,
    fillMissing: true
};

function escapeHTML(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatInlineRichText(value) {
    let safe = escapeHTML(value);
    safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    safe = safe.replace(/__(.+?)__/g, '<strong>$1</strong>');
    return safe;
}

function formatRichTextHTML(text, className = '') {
    const raw = String(text ?? '').replace(/\r\n/g, '\n').trim();
    if (!raw) {
        return `<p class="${className}">No details available.</p>`;
    }

    const listPattern = /^([-*•]\s+|\d+[.)]\s+)/;
    const lines = raw.split('\n').map(line => line.trim()).filter(Boolean);
    if (lines.length > 1 && lines.every(line => listPattern.test(line))) {
        const items = lines.map(line => line.replace(listPattern, '').trim());
        return `<ul class="${className}">${items.map(item => `<li>${formatInlineRichText(item)}</li>`).join('')}</ul>`;
    }

    return `<p class="${className}">${lines.map(line => formatInlineRichText(line)).join('<br>')}</p>`;
}

document.addEventListener('DOMContentLoaded', () => {
    initializeCleaningPage();
});

function initializeCleaningPage() {
    // Get data from sessionStorage
    const statsJson = sessionStorage.getItem('datasetStats');
    const dataJson = sessionStorage.getItem('uploadedFileData');

    if (!statsJson) {
        window.location.href = 'index.html';
        return;
    }

    datasetStats = JSON.parse(statsJson);

    if (dataJson) {
        originalData = JSON.parse(dataJson);
    }

    // Set up event listeners
    const backBtn = document.getElementById('backBtn');
    const confirmCleaning = document.getElementById('confirmCleaning');
    const resetBtn = document.getElementById('resetBtn');
    const downloadBtn = document.getElementById('downloadClearedBtn');
    const viewResultsBtn = document.getElementById('viewResultsBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const proceedBtn = document.getElementById('proceedBtn');
    const closeModal = document.getElementById('closeModal');
    const getAISuggestionsBtn = document.getElementById('getAISuggestions');
    const applySuggestionsBtn = document.getElementById('applySuggestions');

    if (backBtn) {
        backBtn.addEventListener('click', () => {
            window.location.href = 'dataset.html';
        });
    }

    if (confirmCleaning) {
        confirmCleaning.addEventListener('click', () => {
            updateCleaningConfig();
            showConfirmationModal();
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            document.getElementById('removeDuplicates').checked = true;
            document.getElementById('deleteHighMissing').checked = true;
            document.getElementById('deleteLowMissing').checked = true;
            document.getElementById('fillMissing').checked = true;
            updateCleaningConfig();
            displayCleaningSummary();
        });
    }

    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadCleanedData);
    }

    if (viewResultsBtn) {
        viewResultsBtn.addEventListener('click', () => {
            window.location.href = 'cleaning-results.html';
        });
    }

    if (cancelBtn) {
        cancelBtn.addEventListener('click', closeConfirmationModal);
    }

    if (proceedBtn) {
        proceedBtn.addEventListener('click', () => {
            closeConfirmationModal();
            performCleaning();
        });
    }

    if (closeModal) {
        closeModal.addEventListener('click', closeConfirmationModal);
    }

    if (getAISuggestionsBtn) {
        getAISuggestionsBtn.addEventListener('click', getAISuggestions);
    }

    if (applySuggestionsBtn) {
        applySuggestionsBtn.addEventListener('click', applyAISuggestions);
    }

    // Set file name
    document.getElementById('fileName').textContent = `File: ${datasetStats.filename}`;

    // Initialize cleaning info
    updateCleaningConfig();
    displayCleaningSummary();
}

function updateCleaningConfig() {
    cleaningConfig = {
        removeDuplicates: document.getElementById('removeDuplicates').checked,
        deleteHighMissing: document.getElementById('deleteHighMissing').checked,
        deleteLowMissing: document.getElementById('deleteLowMissing').checked,
        fillMissing: document.getElementById('fillMissing').checked
    };
}

function displayCleaningSummary() {
    if (!datasetStats) return;

    // Calculate statistics based on cleaning rules
    const highMissingThreshold = 0.40;
    const lowMissingThreshold = 0.05;

    let columnsToDel = 0;
    let columnsToFill = 0;
    let rowsToDelete = 0;
    let expectedDuplicates = datasetStats.num_duplicates;

    // Analyze each column
    datasetStats.columns.forEach(col => {
        const missingCount = datasetStats.missing_values[col];
        const missingPercent = datasetStats.num_rows > 0 ? missingCount / datasetStats.num_rows : 0;

        if (missingPercent > highMissingThreshold) {
            columnsToDel++;
        } else if (missingPercent > 0 && missingPercent < lowMissingThreshold) {
            // Will delete rows with this column missing
        } else if (missingPercent > 0 && missingPercent >= lowMissingThreshold) {
            columnsToFill++;
        }
    });

    // Display summary
    document.getElementById('columnsToDel').textContent = columnsToDel;
    document.getElementById('columnsToFill').textContent = columnsToFill;
    document.getElementById('duplicatesToRemove').textContent = expectedDuplicates;

    const estimatedRows = datasetStats.num_rows - expectedDuplicates;
    document.getElementById('rowsRemaining').textContent = estimatedRows.toLocaleString();

    // Display treatment plan
    displayTreatmentPlan();
}

function displayTreatmentPlan() {
    if (!datasetStats) return;

    const table = document.getElementById('treatmentTable');
    table.innerHTML = '';

    const highMissingThreshold = 0.40;  // 40%
    const lowMissingThreshold = 0.05;   // 5%

    // Calculate rows with missing values
    let rowsWithMissing = 0;
    const totalRows = datasetStats.num_rows;

    // Estimate rows with missing values (at least one column has missing value)
    // This is an approximation in the frontend - actual calculation is done in backend
    for (let i = 0; i < totalRows; i++) {
        let rowHasMissing = false;
        datasetStats.columns.forEach(col => {
            if (datasetStats.missing_values[col] > 0) {
                // This is a simplified check - just see if any column has missing data
                rowHasMissing = true;
            }
        });
        if (rowHasMissing) rowsWithMissing++;
    }

    // First, show column treatments
    datasetStats.columns.forEach(col => {
        const missingCount = datasetStats.missing_values[col];
        const missingPercent = datasetStats.num_rows > 0 ? (missingCount / datasetStats.num_rows) * 100 : 0;
        const dataType = datasetStats.data_types[col];

        let action = 'Keep';
        let fillMethod = '-';

        if (missingPercent > highMissingThreshold * 100) {
            action = '🗑️ Delete Column';
        } else if (missingPercent > 0 && missingPercent <= lowMissingThreshold * 100) {
            action = '📊 Fill Values';
            // Determine fill method based on AI suggestions or default logic
            if (window.aiFillMethods && col in window.aiFillMethods) {
                fillMethod = window.aiFillMethods[col];
            } else {
                fillMethod = isNumeric(dataType) ? 'Mean/Median' : 'Mode';
            }
        } else if (missingPercent > lowMissingThreshold * 100) {
            action = '📊 Fill Values';
            // Determine fill method based on AI suggestions or default logic
            if (window.aiFillMethods && col in window.aiFillMethods) {
                fillMethod = window.aiFillMethods[col];
            } else {
                fillMethod = isNumeric(dataType) ? 'Mean/Median' : 'Mode';
            }
        }

        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${col}</strong></td>
            <td>${dataType}</td>
            <td>${missingPercent.toFixed(2)}%</td>
            <td>${action}</td>
            <td>${fillMethod}</td>
        `;
        table.appendChild(row);
    });

    // Show row deletion info based on rows with missing values percentage
    const rowsWithMissingPercent = totalRows > 0 ? (rowsWithMissing / totalRows) * 100 : 0;

    if (cleaningConfig.deleteLowMissing && rowsWithMissing > 0) {
        if (rowsWithMissingPercent < 5) {
            // Less than 5% of rows have missing values - suggest deletion
            const summaryRow = document.createElement('tr');
            summaryRow.innerHTML = `
                <td colspan="5" style="text-align: center; padding: 10px; background: rgba(245, 158, 11, 0.14); color: #e2e8f0;"><strong>${rowsWithMissing} rows have missing values (${rowsWithMissingPercent.toFixed(2)}% of total rows) - will be deleted</strong></td>
            `;
            table.appendChild(summaryRow);
        }
    }
}

function isNumeric(dataType) {
    return dataType.includes('int') || dataType.includes('float');
}

function showConfirmationModal() {
    const modal = document.getElementById('confirmationModal');
    const summaryDiv = document.getElementById('confirmationSummary');

    // Build confirmation summary
    let html = '<ul>';

    if (cleaningConfig.removeDuplicates) {
        html += '<li>✓ Remove duplicate rows</li>';
    }

    if (cleaningConfig.deleteHighMissing) {
        html += '<li>✓ Delete columns with >40% missing values</li>';
    }

    if (cleaningConfig.deleteLowMissing) {
        html += '<li>✓ Delete rows with missing values (when <5% of rows are affected)</li>';
    }

    if (cleaningConfig.fillMissing) {
        html += '<li class="warning-item">✓ Fill remaining missing values (mean/median/mode)</li>';
    }

    html += '</ul>';
    html += '<p class="confirmation-warning">⚠️ This will permanently clean your data. Make sure you have reviewed the plan above.</p>';

    summaryDiv.innerHTML = html;
    modal.style.display = 'block';
}

function closeConfirmationModal() {
    const modal = document.getElementById('confirmationModal');
    modal.style.display = 'none';
}

async function performCleaning() {
    updateCleaningConfig();

    document.getElementById('loading').style.display = 'block';
    document.getElementById('errorMessage').style.display = 'none';

    try {
        const response = await fetch(`${API_URL}/clean`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: datasetStats.filename,
                num_rows: datasetStats.num_rows,
                num_columns: datasetStats.num_columns,
                columns: datasetStats.columns,
                data_types: datasetStats.data_types,
                missing_values: datasetStats.missing_values,
                num_duplicates: datasetStats.num_duplicates,
                cleaning_config: cleaningConfig
            })
        });

        const result = await response.json();

        if (result.success) {
            document.getElementById('loading').style.display = 'none';

            // Store all cleaning info in sessionStorage
            sessionStorage.setItem('cleaningResults', JSON.stringify({
                original_stats: datasetStats,
                cleaned_stats: result.cleaned_stats,
                changes_made: result.changes_made,
                sample_data: result.sample_data || [],
                cleaning_config: cleaningConfig
            }));

            // Redirect to results page
            setTimeout(() => {
                window.location.href = 'cleaning-results.html';
            }, 500);
        } else {
            showError(result.error || 'Error cleaning data');
            document.getElementById('loading').style.display = 'none';
        }
    } catch (error) {
        console.error('Cleaning error:', error);
        showError('Error connecting to backend for cleaning');
        document.getElementById('loading').style.display = 'none';
    }
}

function displayCleaningResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';

    // Display before/after stats
    document.getElementById('originalRows').textContent = datasetStats.num_rows.toLocaleString();
    document.getElementById('originalCols').textContent = datasetStats.num_columns;
    document.getElementById('cleanedRows').textContent = result.cleaned_stats.num_rows.toLocaleString();
    document.getElementById('cleanedCols').textContent = result.cleaned_stats.num_columns;

    // Display changes made
    const changesList = document.getElementById('changesList');
    changesList.innerHTML = '';

    if (result.changes_made) {
        result.changes_made.forEach(change => {
            const li = document.createElement('li');
            li.textContent = change;
            changesList.appendChild(li);
        });
    }

    // Store cleaned stats
    sessionStorage.setItem('cleanedDataStats', JSON.stringify(result.cleaned_stats));
}

function downloadCleanedData() {
    const cleaningResults = JSON.parse(sessionStorage.getItem('cleaningResults'));
    if (!cleaningResults || !cleaningResults.cleaned_stats) {
        alert('No cleaned data available');
        return;
    }

    // Request CSV download from backend
    const filename = cleaningResults.cleaned_stats.filename || 'cleaned_data.csv';
    fetch(`${API_URL}/download-cleaned?filename=${encodeURIComponent(filename)}`)
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `cleaned_${filename}`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        })
        .catch(error => {
            console.error('Download error:', error);
            showError('Error downloading cleaned data');
        });
}

async function getAISuggestions() {
    const button = document.getElementById('getAISuggestions');
    const originalText = button.textContent;

    button.textContent = '🤖 Analyzing...';
    button.disabled = true;

    try {
        const response = await fetch(`${API_URL}/cleaning-suggestions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const result = await response.json();

        if (result.success) {
            displayAISuggestions(result);
        } else {
            showError(result.error || 'Error getting AI suggestions');
        }
    } catch (error) {
        console.error('AI suggestions error:', error);
        showError('Error connecting to backend for AI suggestions');
    } finally {
        button.textContent = originalText;
        button.disabled = false;
    }
}

function displayAISuggestions(result) {
    const suggestionsResult = document.getElementById('suggestionsResult');
    const suggestionsList = document.getElementById('suggestionsList');

    suggestionsList.innerHTML = '';

    // Store suggestions and fill methods for later use
    window.aiSuggestions = result.suggestions;
    window.aiFillMethods = result.fill_methods;

    // Display explanations
    result.explanations.forEach(explanation => {
        const div = document.createElement('div');
        div.className = 'suggestion-item';
        div.innerHTML = `<span class="icon">💡</span><span>${formatRichTextHTML(explanation)}</span>`;
        suggestionsList.appendChild(div);
    });

    // Display specific recommendations
    const options = [
        { key: 'removeDuplicates', name: 'Remove Duplicates', icon: '🔄' },
        { key: 'deleteHighMissing', name: 'Delete High Missing Columns', icon: '🗑️' },
        { key: 'deleteLowMissing', name: 'Delete Rows with Missing Values', icon: '📊' },
        { key: 'fillMissing', name: 'Fill Missing Values', icon: '✅' }
    ];

    options.forEach(option => {
        const isRecommended = result.suggestions[option.key];
        const div = document.createElement('div');
        div.className = `suggestion-item ${isRecommended ? 'recommended' : 'not-recommended'}`;
        div.innerHTML = `
            <span class="icon">${option.icon}</span>
            <span><strong>${option.name}:</strong> ${isRecommended ? 'Recommended' : 'Not Recommended'}</span>
        `;
        suggestionsList.appendChild(div);
    });

    // Display fill methods if filling is recommended
    if (result.suggestions.fillMissing && result.fill_methods) {
        const fillMethodsDiv = document.createElement('div');
        fillMethodsDiv.className = 'suggestion-item';
        fillMethodsDiv.innerHTML = '<span class="icon">🔧</span><span><strong>Fill Methods:</strong></span>';
        suggestionsList.appendChild(fillMethodsDiv);

        Object.entries(result.fill_methods).forEach(([col, method]) => {
            const methodDiv = document.createElement('div');
            methodDiv.className = 'suggestion-item fill-method';
            methodDiv.innerHTML = `<span class="icon">•</span><span>Column '${escapeHTML(col)}': ${escapeHTML(method)}</span>`;
            suggestionsList.appendChild(methodDiv);
        });
    }

    suggestionsResult.style.display = 'block';
}

function applyAISuggestions() {
    if (!window.aiSuggestions) {
        alert('No AI suggestions available. Please get suggestions first.');
        return;
    }

    // Apply the suggestions to checkboxes
    document.getElementById('removeDuplicates').checked = window.aiSuggestions.removeDuplicates;
    document.getElementById('deleteHighMissing').checked = window.aiSuggestions.deleteHighMissing;
    document.getElementById('deleteLowMissing').checked = window.aiSuggestions.deleteLowMissing;
    document.getElementById('fillMissing').checked = window.aiSuggestions.fillMissing;

    // Update the cleaning config and display
    updateCleaningConfig();
    displayCleaningSummary();

    // Show success message
    alert('AI suggestions applied! Review the options below and adjust if needed.');
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    }
}
