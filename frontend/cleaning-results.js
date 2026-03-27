// API Configuration - Dynamic for both development and production
function getAPIUrl() {
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return 'http://localhost:8001';
    }
    return window.location.origin;
}

const API_URL = getAPIUrl();
let cleaningResults = null;

document.addEventListener('DOMContentLoaded', () => {
    initializeResultsPage();
});

function initializeResultsPage() {
    // Get data from sessionStorage
    const resultsJson = sessionStorage.getItem('cleaningResults');

    if (!resultsJson) {
        window.location.href = 'cleaning.html';
        return;
    }

    cleaningResults = JSON.parse(resultsJson);

    // Set up event listeners
    document.getElementById('backBtn').addEventListener('click', () => {
        window.location.href = 'cleaning.html';
    });

    document.getElementById('homeBtn').addEventListener('click', () => {
        sessionStorage.clear();
        window.location.href = 'index.html';
    });

    document.getElementById('downloadBtn').addEventListener('click', downloadCleanedData);

    document.getElementById('mlTrainBtn').addEventListener('click', () => {
        // Store cleaned data stats for ML training page
        sessionStorage.setItem('cleanedDataStats', JSON.stringify(cleaningResults.cleaned_stats));
        window.location.href = 'ml-training.html';
    });

    // Display results
    displayResults();
}

function displayResults() {
    if (!cleaningResults) return;

    const originalStats = cleaningResults.original_stats;
    const cleanedStats = cleaningResults.cleaned_stats;
    const changesMade = cleaningResults.changes_made || [];
    const sampleData = cleaningResults.sample_data || [];

    // Set file name
    document.getElementById('fileName').textContent = `File: ${originalStats.filename}`;

    // Display comparison statistics
    document.getElementById('origRows').textContent = originalStats.num_rows.toLocaleString();
    document.getElementById('origCols').textContent = originalStats.num_columns;
    document.getElementById('cleanedRows').textContent = cleanedStats.num_rows.toLocaleString();
    document.getElementById('cleanedCols').textContent = cleanedStats.num_columns;

    // Calculate missing values
    const origMissingSum = Object.values(originalStats.missing_values).reduce((a, b) => a + b, 0);
    const cleanedMissingSum = Object.values(cleanedStats.missing_values || {}).reduce((a, b) => a + b, 0);
    document.getElementById('origMissing').textContent = origMissingSum.toLocaleString();
    document.getElementById('cleanedMissing').textContent = cleanedMissingSum.toLocaleString();

    // Calculate summary statistics
    const rowsRemoved = originalStats.num_rows - cleanedStats.num_rows;
    const colsDeleted = originalStats.num_columns - cleanedStats.num_columns;
    const valuesFilled = origMissingSum - cleanedMissingSum;
    const dataQuality = cleanedStats.num_rows > 0 ?
        Math.round(((1 - (cleanedMissingSum / (cleanedStats.num_rows * cleanedStats.num_columns))) * 100)) : 0;

    document.getElementById('rowsRemoved').textContent = rowsRemoved.toLocaleString();
    document.getElementById('colsDeleted').textContent = colsDeleted;
    document.getElementById('valuesFilled').textContent = valuesFilled.toLocaleString();
    document.getElementById('dataQuality').textContent = dataQuality + '%';

    // Display changes made
    const changesList = document.getElementById('changesList');
    changesList.innerHTML = '';

    if (changesMade.length > 0) {
        changesMade.forEach(change => {
            const li = document.createElement('li');
            li.textContent = change;
            changesList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.textContent = 'Data cleaning completed successfully';
        changesList.appendChild(li);
    }

    // Display sample data
    displaySampleData(sampleData, cleanedStats.columns);
}

function displaySampleData(sampleData, columns) {
    const table = document.getElementById('sampleTable');
    const noDataMsg = document.getElementById('noDataMessage');

    if (!columns || columns.length === 0) {
        table.style.display = 'none';
        noDataMsg.style.display = 'block';
        return;
    }

    // Clear existing content
    table.innerHTML = '';

    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = '<th>#</th>'; // Row number column

    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create body
    const tbody = document.createElement('tbody');

    if (sampleData.length === 0) {
        table.style.display = 'none';
        noDataMsg.style.display = 'block';
        return;
    }

    // Display only first 7 rows
    const displayRows = sampleData.slice(0, 7);

    displayRows.forEach((row, rowIndex) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td style="font-weight: bold; color: #667eea;">${rowIndex + 1}</td>`;

        columns.forEach(col => {
            const td = document.createElement('td');
            const value = row[col];

            if (value === null || value === undefined || value === '') {
                td.textContent = '(empty)';
                td.style.color = '#999';
                td.style.fontStyle = 'italic';
            } else if (typeof value === 'number') {
                td.textContent = typeof value === 'number' ? value.toFixed(2) : value;
                td.style.textAlign = 'right';
            } else {
                td.textContent = String(value).substring(0, 50);
            }

            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    table.style.display = '';
    noDataMsg.style.display = 'none';
}

function downloadCleanedData() {
    if (!cleaningResults || !cleaningResults.cleaned_stats) {
        alert('No cleaned data available to download');
        return;
    }

    const filename = cleaningResults.cleaned_stats.filename || 'cleaned_data.csv';

    fetch(`${API_URL}/download-cleaned?filename=${encodeURIComponent(filename)}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Download failed');
            }
            return response.blob();
        })
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
            alert('Error downloading cleaned data. Please try again.');
        });
}
