// Dataset page specific functionality
const API_URL = 'http://localhost:8001';
let datasetStats = null;

document.addEventListener('DOMContentLoaded', () => {
    displayDatasetDetails();
});

function displayDatasetDetails() {
    const statsJson = sessionStorage.getItem('datasetStats');

    if (!statsJson) {
        console.error('No datasetStats found in sessionStorage');
        // Wait a moment for data to potentially be stored
        setTimeout(() => {
            const retryJson = sessionStorage.getItem('datasetStats');
            if (!retryJson) {
                window.location.href = 'index.html';
            } else {
                datasetStats = JSON.parse(retryJson);
                renderDataset();
            }
        }, 500);
        return;
    }

    try {
        datasetStats = JSON.parse(statsJson);
        console.log('Dataset stats loaded:', datasetStats);
        renderDataset();
    } catch (error) {
        console.error('Error parsing datasetStats:', error);
        window.location.href = 'index.html';
    }
}

function renderDataset() {
    if (!datasetStats) {
        console.error('datasetStats is null');
        return;
    }

    // Display basic info
    displayOverview();
    displayColumnsInfo();
    displayNumericSummary();
    displayCategoricalSummary();
    setupButtons();
}

function setupButtons() {
    const backBtn = document.getElementById('backBtn');
    const cleanBtn = document.getElementById('cleanBtn');

    if (backBtn) {
        backBtn.addEventListener('click', () => {
            sessionStorage.removeItem('datasetStats');
            window.location.href = 'index.html';
        });
    }

    if (cleanBtn) {
        cleanBtn.addEventListener('click', () => {
            window.location.href = 'cleaning.html';
        });
    }
}

function displayOverview() {
    if (!datasetStats) {
        console.error('displayOverview: datasetStats is null');
        return;
    }

    try {
        const fileNameEl = document.getElementById('fileName');
        const totalRowsEl = document.getElementById('totalRows');
        const totalColumnsEl = document.getElementById('totalColumns');
        const missingValuesEl = document.getElementById('missingValues');
        const duplicateRowsEl = document.getElementById('duplicateRows');

        if (fileNameEl) fileNameEl.textContent = `File : ${datasetStats.filename}`;
        if (totalRowsEl) totalRowsEl.textContent = datasetStats.num_rows.toLocaleString();
        if (totalColumnsEl) totalColumnsEl.textContent = datasetStats.num_columns;
        if (missingValuesEl) missingValuesEl.textContent =
            `${datasetStats.total_missing} (${datasetStats.missing_percentage}%)`;
        if (duplicateRowsEl) duplicateRowsEl.textContent = datasetStats.num_duplicates;
    } catch (error) {
        console.error('Error in displayOverview:', error);
    }
}

function displayColumnsInfo() {
    if (!datasetStats) {
        console.error('displayColumnsInfo: datasetStats null');
        return;
    }
    console.log('displayColumnsInfo called');

    const table = document.getElementById('columnsTable');
    if (table) table.innerHTML = '';

    const totalRows = datasetStats.num_rows;

    datasetStats.columns.forEach(col => {
        const dataType = datasetStats.data_types[col];
        const missingCount = datasetStats.missing_values[col];
        const percentage = totalRows > 0 ? ((missingCount / totalRows) * 100).toFixed(2) : '0.00';
        const percentageFill = totalRows > 0 ? (missingCount / totalRows) * 100 : 0;

        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${col}</strong></td>
            <td>${dataType}</td>
            <td><span class="missing-count">${missingCount}</span></td>
            <td>${percentage}%</td>
            <td>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${percentageFill}%"></div>
                </div>
            </td>
        `;
        table.appendChild(row);
    });
}

function displayNumericSummary() {
    console.log('displayNumericSummary called');
    if (!datasetStats || Object.keys(datasetStats.numeric_summary).length === 0) {
        const section = document.getElementById('numericSection');
        if (section) section.style.display = 'none';
        return;
    }

    const table = document.getElementById('numericTable');
    table.innerHTML = '';

    Object.entries(datasetStats.numeric_summary).forEach(([col, stats]) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${col}</strong></td>
            <td>${formatNumber(stats.mean)}</td>
            <td>${formatNumber(stats.median)}</td>
            <td>${formatNumber(stats.std)}</td>
            <td>${formatNumber(stats.min)}</td>
            <td>${formatNumber(stats.max)}</td>
            <td>${stats.missing}</td>
        `;
        table.appendChild(row);
    });
}

function displayCategoricalSummary() {
    console.log('displayCategoricalSummary called');
    if (!datasetStats || Object.keys(datasetStats.categorical_summary).length === 0) {
        const section = document.getElementById('categoricalSection');
        if (section) section.style.display = 'none';
        return;
    }

    const table = document.getElementById('categoricalTable');
    table.innerHTML = '';

    Object.entries(datasetStats.categorical_summary).forEach(([col, stats]) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${col}</strong></td>
            <td>${stats.unique_count}</td>
            <td>${stats.most_common || 'N/A'}</td>
            <td>${stats.missing}</td>
        `;
        table.appendChild(row);
    });
}

// Helper function to format numbers
function formatNumber(num) {
    if (num === null || num === undefined) return 'N/A';
    return Number(num).toFixed(2);
}
