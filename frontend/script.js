// API Configuration
const API_URL = 'http://localhost:8001';
let selectedFile = null;
let datasetStats = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check which page we're on
    const isIndexPage = document.getElementById('uploadBox');

    if (isIndexPage) {
        initializeUploadPage();
    }
});

// ============ UPLOAD PAGE FUNCTIONALITY ============

function initializeUploadPage() {
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const changeFileBtn = document.getElementById('changeFileBtn');

    // Click to browse
    uploadBox.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });

    // Drag over
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });

    // Drag leave
    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });

    // Drop
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Upload button
    uploadBtn.addEventListener('click', () => {
        uploadFile(selectedFile);
    });

    // Change file button
    changeFileBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        document.getElementById('fileInfo').style.display = 'none';
        document.getElementById('errorMessage').style.display = 'none';
        uploadBox.style.display = 'block';
    });
}

function handleFileSelect(file) {
    const validTypes = ['.csv', '.xlsx', '.xls'];
    const fileName = file.name;
    const fileExt = fileName.substring(fileName.lastIndexOf('.')).toLowerCase();

    // Validate file type
    if (!validTypes.includes(fileExt)) {
        showError('Invalid file format. Please upload CSV or Excel file.');
        return;
    }

    // Validate file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
        showError('File size exceeds 50MB limit.');
        return;
    }

    selectedFile = file;
    document.getElementById('fileName').textContent = fileName;
    document.getElementById('uploadBox').style.display = 'none';
    document.getElementById('fileInfo').style.display = 'block';
    document.getElementById('errorMessage').style.display = 'none';
}

async function uploadFile(file) {
    if (!file) {
        showError('Please select a file first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('fileInfo').style.display = 'none';

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            // Store the stats in sessionStorage for the dataset page
            sessionStorage.setItem('datasetStats', JSON.stringify(data));
            // Redirect to dataset page
            window.location.href = 'dataset.html';
        } else {
            showError(data.error || 'Error uploading file');
            document.getElementById('loading').style.display = 'none';
            document.getElementById('fileInfo').style.display = 'block';
        }
    } catch (error) {
        console.error('Upload error:', error);
        showError('Error connecting to server. Make sure the backend is running on http://localhost:8001');
        document.getElementById('loading').style.display = 'none';
        document.getElementById('fileInfo').style.display = 'block';
    }
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    }
}
