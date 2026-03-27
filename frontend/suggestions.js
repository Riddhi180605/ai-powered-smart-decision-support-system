// API Configuration
const API_URL = 'http://localhost:8001';
let mlResults = null;
let suggestionsData = null;
let aiInsightsData = null;
let ragIndexReady = false;
let whatIfChanges = {};
let whatIfConfig = null;
let lastRegularTab = 'features';

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

function formatRichTextHTML(text, className = 'suggestion-text') {
    const raw = String(text ?? '').replace(/\r\n/g, '\n').trim();
    if (!raw) {
        return `<p class="${className}">No details available.</p>`;
    }

    const listPattern = /^([-*•]\s+|\d+[.)]\s+)/;
    const blocks = raw.split(/\n{2,}/).map(b => b.trim()).filter(Boolean);

    return blocks.map(block => {
        const lines = block.split('\n').map(line => line.trim()).filter(Boolean);
        if (lines.length > 1 && lines.every(line => listPattern.test(line))) {
            const items = lines.map(line => line.replace(listPattern, '').trim());
            return `<ul class="${className}">${items.map(item => `<li>${formatInlineRichText(item)}</li>`).join('')}</ul>`;
        }

        return `<p class="${className}">${lines.map(line => formatInlineRichText(line)).join('<br>')}</p>`;
    }).join('');
}

document.addEventListener('DOMContentLoaded', () => {
    initializeSuggestionsPage();
});

function initializeSuggestionsPage() {
    // Get ML results from sessionStorage
    const mlResultsJson = sessionStorage.getItem('mlResults');

    if (!mlResultsJson) {
        showError('No ML training results available. Please train a model first.');
        return;
    }

    mlResults = JSON.parse(mlResultsJson);

    // Set up event listeners
    const backBtn = document.getElementById('backBtn');
    if (backBtn) {
        backBtn.addEventListener('click', () => {
            window.location.href = 'ml-training.html';
        });
    }

    // AI Insights button
    const generateAIInsightsBtn = document.getElementById('generateAIInsightsBtn');
    if (generateAIInsightsBtn) {
        generateAIInsightsBtn.addEventListener('click', generateAIInsights);
    }

    // RAG chatbot controls
    const buildRAGIndexBtn = document.getElementById('buildRAGIndexBtn');
    if (buildRAGIndexBtn) {
        buildRAGIndexBtn.addEventListener('click', buildRAGIndex);
    }

    const sendRAGQueryBtn = document.getElementById('sendRAGQueryBtn');
    if (sendRAGQueryBtn) {
        sendRAGQueryBtn.addEventListener('click', askRAGQuestion);
    }

    const ragQueryInput = document.getElementById('ragQueryInput');
    if (ragQueryInput) {
        ragQueryInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                askRAGQuestion();
            }
        });
    }

    const askSampleProfitBtn = document.getElementById('askSampleProfitBtn');
    if (askSampleProfitBtn) {
        askSampleProfitBtn.addEventListener('click', () => askSampleQuestion('Why did profit decrease last month?'));
    }

    const askSampleChurnBtn = document.getElementById('askSampleChurnBtn');
    if (askSampleChurnBtn) {
        askSampleChurnBtn.addEventListener('click', () => askSampleQuestion('Which feature impacts churn most?'));
    }

    const askSampleRegionBtn = document.getElementById('askSampleRegionBtn');
    if (askSampleRegionBtn) {
        askSampleRegionBtn.addEventListener('click', () => askSampleQuestion('Show trends in Region-1'));
    }

    // What-if simulation controls
    const addWhatIfChangeBtn = document.getElementById('addWhatIfChangeBtn');
    if (addWhatIfChangeBtn) {
        addWhatIfChangeBtn.addEventListener('click', addWhatIfChange);
    }

    const runWhatIfBtn = document.getElementById('runWhatIfBtn');
    if (runWhatIfBtn) {
        runWhatIfBtn.addEventListener('click', runWhatIfSimulation);
    }

    const clearWhatIfChangesBtn = document.getElementById('clearWhatIfChangesBtn');
    if (clearWhatIfChangesBtn) {
        clearWhatIfChangesBtn.addEventListener('click', clearWhatIfChanges);
    }

    const whatIfMode = document.getElementById('whatIfMode');
    if (whatIfMode) {
        whatIfMode.addEventListener('change', () => {
            updateWhatIfValueHint();
            updateWhatIfValueInputMode();
        });
    }

    const whatIfFeature = document.getElementById('whatIfFeature');
    if (whatIfFeature) {
        whatIfFeature.addEventListener('change', () => {
            updateWhatIfValueInputMode();
        });
    }

    const whatIfRowIndex = document.getElementById('whatIfRowIndex');
    if (whatIfRowIndex) {
        whatIfRowIndex.addEventListener('change', loadWhatIfConfig);
    }

    const presetWrap = document.getElementById('whatIfQuickPresets');
    if (presetWrap) {
        presetWrap.addEventListener('click', onWhatIfPresetClick);
    }

    initializeWhatIfFeatureOptions();
    updateWhatIfValueHint();
    updateWhatIfValueInputMode();
    renderWhatIfChanges();
    loadWhatIfConfig();

    // Generate AI suggestions
    generateAISuggestions();
}

async function generateAISuggestions() {
    try {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('contentContainer').style.display = 'none';
        document.getElementById('errorMessage').style.display = 'none';

        console.log('Requesting AI suggestions from backend...');

        const response = await fetch(`${API_URL}/get-ai-suggestions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                target_column: mlResults.target_column,
                problem_type: mlResults.problem_type,
                best_model: mlResults.best_model,
                results: mlResults.results,
                dataset_info: mlResults.dataset_info
            })
        });

        const result = await response.json();

        if (result.success) {
            suggestionsData = result;
            displayAllSuggestions(result);
        } else {
            showError(result.error || 'Error generating AI suggestions');
        }
    } catch (error) {
        console.error('Suggestions generation error:', error);
        showError('Error connecting to backend for AI suggestions');
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function displayAllSuggestions(data) {
    try {
        // Problem Type Analysis
        displayProblemTypeAnalysis(data);

        // Feature Impact
        displayFeatureImpact(data);

        // Improvement Recommendations
        displayImprovementRecommendations(data);

        // Key Insights
        displayKeyInsights(data);

        // Model Selection Reasoning
        displayModelSelection(data);

        // Data Suggestions
        displayDataSuggestions(data);

        // General Recommendations
        displayGeneralRecommendations(data);

        // Show content
        document.getElementById('contentContainer').style.display = 'block';
    } catch (error) {
        console.error('Error displaying suggestions:', error);
        showError('Error displaying suggestions: ' + error.message);
    }
}

function displayProblemTypeAnalysis(data) {
    const container = document.getElementById('problemTypeAnalysis');

    const problemType = data.problem_type || mlResults.problem_type;
    const analysis = data.problem_type_analysis || {};

    let html = `
        <h4>Problem Type: <strong>${problemType === 'classification' ? 'Classification' : 'Regression'}</strong></h4>
        <p class="suggestion-text">${analysis.explanation || 'Default explanation'}</p>
        
        <h4>What This Means:</h4>
        <ul class="score-explanation-list">
    `;

    if (problemType === 'classification') {
        html += `
            <li>Your task is to predict categories/classes for the target column</li>
            <li>The model learns to categorize inputs into predefined groups</li>
            <li>Common metrics: Accuracy, Precision, Recall, F1-Score</li>
            <li>Best for: Yes/No decisions, categories, labels</li>
        `;
    } else {
        html += `
            <li>Your task is to predict continuous numerical values for the target column</li>
            <li>The model learns to estimate specific values based on input features</li>
            <li>Common metrics: RMSE (Root Mean Squared Error), R² Score</li>
            <li>Best for: Price predictions, quantity forecasting, value estimation</li>
        `;
    }

    html += `</ul>`;

    container.innerHTML = html;
}

function displayFeatureImpact(data) {
    const features = data.feature_importance || [];

    // Display top feature
    if (features.length > 0) {
        document.getElementById('topFeature').textContent = features[0].feature;
        document.getElementById('topFeatureScore').textContent = (features[0].importance * 100).toFixed(1) + '%';
    }

    // Display feature list
    const listContainer = document.getElementById('featureImportanceList');
    if (listContainer && features.length > 0) {
        let html = '<h4>Top 10 Most Impactful Features:</h4>';

        features.slice(0, 10).forEach((feature, index) => {
            const importance = feature.importance * 100;
            let impactLevel = 'low';
            if (importance > 20) impactLevel = 'high';
            else if (importance > 10) impactLevel = 'medium';

            html += `
                <div class="feature-recommendation">
                    <div class="feature-name">${index + 1}. ${feature.feature}</div>
                    <div class="recommendation-text">
                        Impact: <span class="impact-badge impact-${impactLevel}">${importance.toFixed(1)}%</span>
                    </div>
                    <div class="recommendation-text">
                        This feature has ${impactLevel} impact on your model's predictions.
                    </div>
                </div>
            `;
        });

        listContainer.innerHTML = html;
    }
}

function displayImprovementRecommendations(data) {
    const recommendations = data.feature_recommendations || [];
    const container = document.getElementById('improvementRecommendations');

    if (!container || recommendations.length === 0) {
        return;
    }

    let html = '';

    recommendations.forEach((rec, index) => {
        const direction = rec.direction === 'increase' ? 'up' : 'down';
        const directionText = rec.direction === 'increase' ? '⬆️ INCREASE' : '⬇️ DECREASE';

        html += `
            <div class="feature-recommendation">
                <div class="feature-name">${index + 1}. ${rec.feature}</div>
                <div class="recommendation-text">
                    <strong>Recommendation:</strong> <span class="direction-badge direction-${direction}">${directionText}</span>
                </div>
                <div class="recommendation-text">
                    <strong>Reason:</strong> ${rec.reason}
                </div>
                <div class="recommendation-text">
                    <strong>Impact:</strong> This change will help improve your "${mlResults.target_column}" column.
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

function displayKeyInsights(data) {
    const insights = data.key_insights || [];
    const container = document.getElementById('keyInsights');

    if (!container || insights.length === 0) {
        return;
    }

    let html = '';

    insights.forEach((insight) => {
        html += `
            <div class="feature-recommendation">
                ${formatRichTextHTML(insight, 'recommendation-text')}
            </div>
        `;
    });

    container.innerHTML = html;
}

function displayModelSelection(data) {
    const modelReason = data.model_selection_reason || {};

    document.getElementById('selectedModel').textContent = mlResults.best_model || 'Unknown';

    const bestModelMetrics = mlResults.results[mlResults.best_model] || {};

    // Calculate score based on problem type
    let score = 0;
    if (mlResults.problem_type === 'classification') {
        score = (bestModelMetrics.f1_score * 100).toFixed(1);
    } else {
        score = (bestModelMetrics.r2_score * 100).toFixed(1);
    }

    document.getElementById('modelScore').textContent = score + '%';

    // Display reasons
    const reasonsContainer = document.getElementById('modelReasons');
    let reasonsHtml = '';

    const reasons = modelReason.reasons || [];
    reasons.forEach((reason) => {
        reasonsHtml += `<li>${reason}</li>`;
    });

    // Add default reasons if none provided
    if (reasonsHtml === '') {
        if (mlResults.problem_type === 'classification') {
            reasonsHtml += `
                <li>Highest F1-Score (${bestModelMetrics.f1_score?.toFixed(4) || 'N/A'}) among all models</li>
                <li>Best balance between precision and recall</li>
                <li>Consistent cross-validation performance</li>
            `;
        } else {
            reasonsHtml += `
                <li>Lowest RMSE (${bestModelMetrics.rmse?.toFixed(4) || 'N/A'}) among all models</li>
                <li>Highest R² Score (${bestModelMetrics.r2_score?.toFixed(4) || 'N/A'})</li>
                <li>Best generalization on test data</li>
            `;
        }
    }

    reasonsContainer.innerHTML = reasonsHtml;

    // Display model comparison
    displayModelComparison(data);
}

function displayModelComparison(data) {
    const container = document.getElementById('modelComparison');
    const results = mlResults.results || {};

    let html = '<table class="model-compare-table">';
    html += '<tr class="model-compare-header">';

    if (mlResults.problem_type === 'classification') {
        html += '<th>Model</th>';
        html += '<th>Accuracy</th>';
        html += '<th>F1-Score</th>';
        html += '<th>Status</th>';
        html += '</tr>';

        Object.entries(results).forEach(([modelName, metrics]) => {
            const rowClass = modelName === mlResults.best_model ? 'model-selected-row' : '';
            html += `<tr class="${rowClass}">`;
            html += `<td><strong>${modelName}</strong>${modelName === mlResults.best_model ? ' 🏆' : ''}</td>`;

            if ('error' in metrics) {
                html += `<td colspan="2" class="model-error-cell">❌ ${metrics.error}</td>`;
                html += '<td class="model-status-cell">Failed</td>';
            } else {
                html += `<td class="model-status-cell">${metrics.accuracy?.toFixed(4)}</td>`;
                html += `<td class="model-status-cell">${metrics.f1_score?.toFixed(4)}</td>`;
                html += '<td class="model-status-cell">✅ Trained</td>';
            }

            html += '</tr>';
        });
    } else {
        html += '<th>Model</th>';
        html += '<th>RMSE</th>';
        html += '<th>R² Score</th>';
        html += '<th>Status</th>';
        html += '</tr>';

        Object.entries(results).forEach(([modelName, metrics]) => {
            const rowClass = modelName === mlResults.best_model ? 'model-selected-row' : '';
            html += `<tr class="${rowClass}">`;
            html += `<td><strong>${modelName}</strong>${modelName === mlResults.best_model ? ' 🏆' : ''}</td>`;

            if ('error' in metrics) {
                html += `<td colspan="2" class="model-error-cell">❌ ${metrics.error}</td>`;
                html += '<td class="model-status-cell">Failed</td>';
            } else {
                html += `<td class="model-status-cell">${metrics.rmse?.toFixed(4)}</td>`;
                html += `<td class="model-status-cell">${metrics.r2_score?.toFixed(4)}</td>`;
                html += '<td class="model-status-cell">✅ Trained</td>';
            }

            html += '</tr>';
        });
    }

    html += '</table>';
    container.innerHTML = html;
}

function displayDataSuggestions(data) {
    const container = document.getElementById('dataSuggestions');
    const suggestions = data.data_suggestions || [];

    if (!container || suggestions.length === 0) {
        return;
    }

    let html = '';

    suggestions.forEach((suggestion) => {
        html += `
            <div class="feature-recommendation">
                <div class="recommendation-text">
                    <strong>Suggestion:</strong> ${suggestion}
                </div>
            </div>
        `;
    });

    container.innerHTML = html;

    // Display dataset status
    const statusContainer = document.getElementById('datasetStatus');
    if (statusContainer) {
        const info = mlResults.dataset_info || {};

        let statusHtml = `
            <div class="summary-box">
                <h4>Total Samples</h4>
                <div class="summary-value">${info.total_samples?.toLocaleString() || 'N/A'}</div>
            </div>
            <div class="summary-box">
                <h4>Features</h4>
                <div class="summary-value">${info.features || 'N/A'}</div>
            </div>
            <div class="summary-box">
                <h4>Train Samples</h4>
                <div class="summary-value">${info.train_samples?.toLocaleString() || 'N/A'}</div>
            </div>
            <div class="summary-box">
                <h4>Test Samples</h4>
                <div class="summary-value">${info.test_samples?.toLocaleString() || 'N/A'}</div>
            </div>
        `;

        statusContainer.innerHTML = statusHtml;
    }
}

function displayGeneralRecommendations(data) {
    const container = document.getElementById('generalRecommendations');
    const recommendations = data.general_recommendations || [];

    if (!container || recommendations.length === 0) {
        return;
    }

    let html = '<ul class="score-explanation-list">';

    recommendations.forEach((rec) => {
        html += `<li>${rec}</li>`;
    });

    html += '</ul>';

    container.innerHTML = html;
}

function switchTab(tabName) {
    const isWhatIf = tabName === 'what-if';

    if (!isWhatIf) {
        lastRegularTab = tabName;
    }

    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));

    // Remove active class from all buttons
    const buttons = document.querySelectorAll('.tab-button');
    buttons.forEach(btn => btn.classList.remove('active'));

    // Show selected tab
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }

    // Enable centered, focused mode when What-If is open.
    document.body.classList.toggle('whatif-focus', isWhatIf);

    // Add active class to clicked tab button when available.
    if (typeof event !== 'undefined' && event.target && event.target.classList && event.target.classList.contains('tab-button')) {
        event.target.classList.add('active');
        return;
    }

    // Fallback for programmatic tab switches or non-tab-button triggers.
    buttons.forEach(btn => {
        const clickHandler = btn.getAttribute('onclick') || '';
        if (clickHandler.includes(`'${tabName}'`) || clickHandler.includes(`"${tabName}"`)) {
            btn.classList.add('active');
        }
    });
}

function closeWhatIfFocus() {
    switchTab(lastRegularTab || 'features');
}

async function generateAIInsights() {
    try {
        const btn = document.getElementById('generateAIInsightsBtn');
        const originalText = btn.textContent;
        btn.textContent = '🔄 Generating...';
        btn.disabled = true;

        console.log('Requesting AI-powered insights from backend...');

        const response = await fetch(`${API_URL}/generate-ai-insights`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                target_column: mlResults.target_column,
                dataset_type: suggestionsData ? suggestionsData.dataset_type : 'General',
                include_historical_analysis: true
            })
        });

        const result = await response.json();

        if (result.success) {
            aiInsightsData = result;
            displayAIInsights(result);
            document.getElementById('aiInsightsContent').style.display = 'block';
        } else {
            showError(result.error || 'Error generating AI insights');
        }
    } catch (error) {
        console.error('AI Insights generation error:', error);
        showError('Error connecting to backend for AI insights. Make sure your LLM API key is set (for example GROQ_API_KEY).');
    } finally {
        const btn = document.getElementById('generateAIInsightsBtn');
        btn.textContent = '✅ Insights Generated';
        btn.disabled = true; // Keep disabled after success
    }
}

function displayAIInsights(data) {
    try {
        // Prediction Interpretation
        displayPredictionInterpretation(data);

        // Feature Analysis
        displayFeatureAnalysis(data);

        // Trend Comparison
        displayTrendComparison(data);

        // Actionable Recommendations
        displayActionableRecommendations(data);

        // Risk Assessment (if available)
        if (data.risk_assessment) {
            displayRiskAssessment(data.risk_assessment);
            document.getElementById('riskAssessmentSection').style.display = 'block';
        }

        // Implementation Roadmap (if available)
        if (data.implementation_roadmap) {
            displayImplementationRoadmap(data.implementation_roadmap);
            document.getElementById('roadmapSection').style.display = 'block';
        }

    } catch (error) {
        console.error('Error displaying AI insights:', error);
        showError('Error displaying AI insights');
    }
}

function displayPredictionInterpretation(data) {
    const container = document.getElementById('predictionInterpretation');
    if (!container || !data.insights) return;

    const predictionInsight = data.insights.find(i => i.type === 'prediction_interpretation');
    if (predictionInsight) {
        container.innerHTML = formatRichTextHTML(predictionInsight.content);
    } else {
        container.innerHTML = '<p class="suggestion-text">No prediction interpretation available.</p>';
    }
}

function displayFeatureAnalysis(data) {
    const container = document.getElementById('featureAnalysis');
    if (!container || !data.insights) return;

    const featureInsight = data.insights.find(i => i.type === 'feature_analysis');
    if (featureInsight) {
        container.innerHTML = formatRichTextHTML(featureInsight.content);
    } else {
        container.innerHTML = '<p class="suggestion-text">No feature analysis available.</p>';
    }

    // Also show top features
    if (data.feature_importance && data.feature_importance.length > 0) {
        let html = '<h4>Top Predictive Features:</h4><ul>';
        data.feature_importance.slice(0, 5).forEach(feature => {
            html += `<li><strong>${feature.feature}</strong> (Impact: ${(feature.importance * 100).toFixed(1)}%)</li>`;
        });
        html += '</ul>';
        container.innerHTML += html;
    }
}

function displayTrendComparison(data) {
    const container = document.getElementById('trendComparison');
    if (!container || !data.insights) return;

    const trendInsight = data.insights.find(i => i.type === 'trend_comparison');
    if (trendInsight) {
        container.innerHTML = formatRichTextHTML(trendInsight.content);
    } else {
        container.innerHTML = '<p class="suggestion-text">No trend comparison available.</p>';
    }

    // Add trend analysis if available
    if (data.trend_analysis) {
        const trend = data.trend_analysis;
        let trendHtml = '<div class="insight-stat-box">';
        trendHtml += `<p><strong>Trend Direction:</strong> ${trend.trend_direction || 'Stable'}</p>`;
        if (trend.historical_avg !== undefined && trend.current_avg !== undefined) {
            trendHtml += `<p><strong>Historical Average:</strong> ${trend.historical_avg.toFixed(2)}</p>`;
            trendHtml += `<p><strong>Current Average:</strong> ${trend.current_avg.toFixed(2)}</p>`;
        }
        trendHtml += '</div>';
        container.innerHTML += trendHtml;
    }
}

function displayActionableRecommendations(data) {
    const container = document.getElementById('actionableRecommendations');
    if (!container || !data.insights) return;

    const recommendationsInsight = data.insights.find(i => i.type === 'actionable_recommendations');
    if (recommendationsInsight) {
        container.innerHTML = formatRichTextHTML(recommendationsInsight.content);
    } else {
        container.innerHTML = '<p class="suggestion-text">No actionable recommendations available.</p>';
    }
}

function displayRiskAssessment(riskData) {
    const container = document.getElementById('riskAssessment');
    if (!container) return;

    let html = `<div class="impact-badge impact-${riskData.risk_level.toLowerCase()}">${riskData.risk_level} Risk</div>`;
    html += '<h4>Risk Factors:</h4><ul>';

    riskData.risk_factors.forEach(factor => {
        html += `<li>${factor}</li>`;
    });

    html += '</ul><h4>Mitigation Suggestions:</h4><ul>';
    riskData.mitigation_suggestions.forEach(suggestion => {
        html += `<li>${suggestion}</li>`;
    });

    html += '</ul>';
    container.innerHTML = html;
}

function displayImplementationRoadmap(roadmapData) {
    const container = document.getElementById('implementationRoadmap');
    if (!container) return;

    let html = '';
    roadmapData.forEach(phase => {
        html += `
            <div class="roadmap-phase">
                <h4>${phase.phase}</h4>
                <p><strong>Timeline:</strong> ${phase.timeline}</p>
                <p><strong>Resources Needed:</strong> ${phase.resources_needed.join(', ')}</p>
                <h5>Actions:</h5>
                <ul>
                    ${phase.actions.map(action => `<li>${action}</li>`).join('')}
                </ul>
            </div>
        `;
    });

    container.innerHTML = html;
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    }

    document.getElementById('loading').style.display = 'none';
    document.getElementById('contentContainer').style.display = 'none';
}

function appendChatMessage(role, text) {
    const log = document.getElementById('ragChatLog');
    if (!log) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    if (role === 'user') {
        messageDiv.innerHTML = `<p class="suggestion-text">${formatInlineRichText(text)}</p>`;
    } else {
        messageDiv.innerHTML = formatRichTextHTML(text);
    }
    log.appendChild(messageDiv);
    log.scrollTop = log.scrollHeight;
}

function setRAGStatus(text) {
    const status = document.getElementById('ragStatus');
    if (status) {
        status.textContent = text;
    }
}

function setRAGButtonsDisabled(disabled) {
    const buildBtn = document.getElementById('buildRAGIndexBtn');
    const sendBtn = document.getElementById('sendRAGQueryBtn');
    if (buildBtn) buildBtn.disabled = disabled;
    if (sendBtn) sendBtn.disabled = disabled;
}

async function buildRAGIndex() {
    try {
        setRAGButtonsDisabled(true);
        setRAGStatus('Index status: Building...');
        appendChatMessage('system', 'Building vector index from historical dataset, ML outputs, and SHAP explanations...');

        const response = await fetch(`${API_URL}/chatbot/build-index`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                embedding_model: 'all-MiniLM-L6-v2'
            })
        });

        const result = await response.json();

        if (result.success) {
            ragIndexReady = true;
            const details = result.index_details || {};
            setRAGStatus(`Index status: Ready (${details.num_chunks || 0} chunks)`);
            appendChatMessage('system', 'RAG index built successfully. Ask your question now.');
            return true;
        } else {
            ragIndexReady = false;
            setRAGStatus('Index status: Failed to build');
            appendChatMessage('system', `Failed to build index: ${result.error || 'Unknown error'}`);
            return false;
        }
    } catch (error) {
        ragIndexReady = false;
        setRAGStatus('Index status: Failed to build');
        appendChatMessage('system', `Error while building index: ${error.message}`);
        return false;
    } finally {
        setRAGButtonsDisabled(false);
    }
}

function renderRetrievedContext(items) {
    const container = document.getElementById('ragRetrievedContext');
    if (!container) return;

    if (!items || items.length === 0) {
        container.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    let html = '<h4>Retrieved Context Chunks</h4>';
    items.forEach((item) => {
        const score = typeof item.score === 'number' ? item.score.toFixed(4) : 'N/A';
        html += `
            <details>
                <summary>${item.source || 'unknown_source'} | score=${score}</summary>
                ${formatRichTextHTML(item.text || '')}
            </details>
        `;
    });

    container.innerHTML = html;
    container.style.display = 'block';
}

async function askRAGQuestion() {
    const input = document.getElementById('ragQueryInput');
    const query = input ? input.value.trim() : '';

    if (!query) {
        appendChatMessage('system', 'Please enter a question first.');
        return;
    }

    appendChatMessage('user', query);
    if (input) input.value = '';

    try {
        setRAGButtonsDisabled(true);

        if (!ragIndexReady) {
            const built = await buildRAGIndex();
            if (!built) {
                appendChatMessage('system', 'Cannot answer yet because index build failed. Please fix the error and try again.');
                return;
            }
        }

        appendChatMessage('system', 'Retrieving relevant context and generating answer...');

        const response = await fetch(`${API_URL}/chatbot/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query })
        });

        const result = await response.json();

        if (result.success) {
            appendChatMessage('assistant', result.answer || 'No answer generated.');
            renderRetrievedContext(result.retrieved_context || []);
        } else {
            appendChatMessage('system', `Chatbot error: ${result.error || result.answer || 'Unknown error'}`);
        }
    } catch (error) {
        appendChatMessage('system', `Error asking question: ${error.message}`);
    } finally {
        setRAGButtonsDisabled(false);
    }
}

function askSampleQuestion(questionText) {
    const input = document.getElementById('ragQueryInput');
    if (input) {
        input.value = questionText;
    }
    askRAGQuestion();
}

function initializeWhatIfFeatureOptions() {
    const featureSelect = document.getElementById('whatIfFeature');
    if (!featureSelect) return;

    const featureColumns = (whatIfConfig && whatIfConfig.feature_columns) ||
        (mlResults && mlResults.dataset_info && mlResults.dataset_info.feature_columns) || [];
    const featureTypes = (whatIfConfig && whatIfConfig.feature_types) || {};

    featureSelect.innerHTML = '';

    if (!featureColumns.length) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No features available';
        featureSelect.appendChild(option);
        return;
    }

    featureColumns.forEach((feature) => {
        const option = document.createElement('option');
        option.value = feature;
        const typeTag = featureTypes[feature] === 'categorical' ? ' (text)' : '';
        option.textContent = `${feature}${typeTag}`;
        featureSelect.appendChild(option);
    });
}

async function loadWhatIfConfig() {
    const rowIndexEl = document.getElementById('whatIfRowIndex');
    const rowText = rowIndexEl ? rowIndexEl.value.trim() : '';
    const payload = {};
    if (rowText !== '') {
        payload.row_index = Number(rowText);
    }

    try {
        const response = await fetch(`${API_URL}/what-if-config`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (!result.success) {
            return;
        }

        whatIfConfig = result;
        initializeWhatIfFeatureOptions();
        updateWhatIfValueInputMode();
        renderWhatIfRowPreview(result);

        if (rowIndexEl && rowIndexEl.value.trim() === '') {
            rowIndexEl.value = String(result.default_row_index ?? 0);
        }

        const helpEl = document.getElementById('whatIfRowHelp');
        if (helpEl) {
            helpEl.textContent = `Row index starts at 0. Total rows: ${result.row_count}. Using row ${result.default_row_index}.`;
        }
    } catch (error) {
        console.error('Failed to load what-if config:', error);
    }
}

function renderWhatIfRowPreview(config) {
    const previewEl = document.getElementById('whatIfRowPreview');
    if (!previewEl) return;

    const rowPreview = config && config.row_preview ? config.row_preview : {};
    const entries = Object.entries(rowPreview).slice(0, 8);
    if (!entries.length) {
        previewEl.style.display = 'none';
        previewEl.innerHTML = '';
        return;
    }

    const parts = entries.map(([k, v]) => `<div class="recommendation-text"><strong>${escapeHTML(k)}:</strong> ${escapeHTML(v)}</div>`).join('');
    previewEl.innerHTML = `<h4>Selected Row Preview (first columns)</h4>${parts}`;
    previewEl.style.display = 'block';
}

function updateWhatIfValueInputMode() {
    const featureEl = document.getElementById('whatIfFeature');
    const modeEl = document.getElementById('whatIfMode');
    const valueInput = document.getElementById('whatIfValue');
    const categorySelect = document.getElementById('whatIfCategoryValue');
    const modeHelp = document.getElementById('whatIfModeHelp');
    if (!featureEl || !modeEl || !valueInput || !categorySelect) return;

    const feature = featureEl.value;
    const isCategorical = whatIfConfig && whatIfConfig.feature_types && whatIfConfig.feature_types[feature] === 'categorical';

    if (isCategorical) {
        modeEl.value = 'set';
        modeEl.disabled = true;
        valueInput.style.display = 'none';
        categorySelect.style.display = 'block';
        categorySelect.innerHTML = '';

        const allowed = (whatIfConfig.category_options && whatIfConfig.category_options[feature]) || [];
        allowed.forEach((item) => {
            const option = document.createElement('option');
            option.value = item;
            option.textContent = item;
            categorySelect.appendChild(option);
        });

        if (modeHelp) {
            modeHelp.textContent = 'This feature is text/categorical. Use "Set exact value" with one of dataset categories.';
        }
    } else {
        modeEl.disabled = false;
        valueInput.style.display = 'block';
        categorySelect.style.display = 'none';
        if (modeHelp) {
            modeHelp.textContent = 'Numeric feature: you can set exact value, add/subtract amount, or change by %.';
        }
    }
}

function addWhatIfChange() {
    const featureEl = document.getElementById('whatIfFeature');
    const modeEl = document.getElementById('whatIfMode');
    const valueEl = document.getElementById('whatIfValue');
    const categoryValueEl = document.getElementById('whatIfCategoryValue');

    const feature = featureEl ? featureEl.value : '';
    const mode = modeEl ? modeEl.value : 'set';
    const isCategorical = whatIfConfig && whatIfConfig.feature_types && whatIfConfig.feature_types[feature] === 'categorical';
    const rawValue = isCategorical
        ? (categoryValueEl ? String(categoryValueEl.value).trim() : '')
        : (valueEl ? valueEl.value.trim() : '');

    if (!feature) {
        alert('Select a feature first.');
        return;
    }

    if (!rawValue) {
        alert('Enter a change value.');
        return;
    }

    let parsedValue = rawValue;
    if (isCategorical) {
        parsedValue = rawValue;
    } else if (mode !== 'set') {
        const normalized = rawValue.replace('%', '').trim();
        const numericValue = Number(normalized);
        if (!Number.isFinite(numericValue)) {
            alert('For amount/% mode, use a numeric value like 10, -5, +20.');
            return;
        }
        parsedValue = numericValue;
    }

    whatIfChanges[feature] = {
        mode,
        value: parsedValue
    };

    if (valueEl) {
        valueEl.value = '';
    }

    renderWhatIfChanges();
}

function clearWhatIfChanges() {
    whatIfChanges = {};
    renderWhatIfChanges();
}

function updateWhatIfValueHint() {
    const modeEl = document.getElementById('whatIfMode');
    const valueEl = document.getElementById('whatIfValue');
    if (!modeEl || !valueEl) return;

    if (modeEl.value === 'set') {
        valueEl.placeholder = 'Example: 50';
    } else if (modeEl.value === 'add') {
        valueEl.placeholder = 'Example: +5 or -3';
    } else {
        valueEl.placeholder = 'Example: +20 or -10';
    }
}

function onWhatIfPresetClick(event) {
    const target = event.target;
    if (!target || !target.dataset || !target.dataset.preset) return;

    const preset = target.dataset.preset;
    const [mode, value] = preset.split(':');

    const modeEl = document.getElementById('whatIfMode');
    const valueEl = document.getElementById('whatIfValue');
    if (!modeEl || !valueEl) return;

    modeEl.value = mode;
    valueEl.value = value;
    updateWhatIfValueHint();
    updateWhatIfValueInputMode();
}

function removeWhatIfChange(feature) {
    delete whatIfChanges[feature];
    renderWhatIfChanges();
}

function renderWhatIfChanges() {
    const list = document.getElementById('whatIfChangeList');
    if (!list) return;

    const entries = Object.entries(whatIfChanges);
    if (!entries.length) {
        list.innerHTML = '<div class="recommendation-text">No changes added yet.</div>';
        return;
    }

    let html = '';
    const modeText = {
        set: 'Set',
        add: 'Amount',
        percent: 'Percent'
    };

    entries.forEach(([feature, payload]) => {
        html += `
            <div class="change-item">
                <div><strong>${escapeHTML(feature)}</strong> → ${modeText[payload.mode] || payload.mode}: <strong>${escapeHTML(payload.value)}</strong></div>
                <button type="button" onclick="removeWhatIfChange('${feature.replace(/'/g, "\\'")}')">Remove</button>
            </div>
        `;
    });

    list.innerHTML = html;
}

async function runWhatIfSimulation() {
    try {
        const changes = { ...whatIfChanges };
        if (!Object.keys(changes).length) {
            alert('Add at least one feature change before running simulation.');
            return;
        }

        const rowIndexEl = document.getElementById('whatIfRowIndex');
        const rowIndexText = rowIndexEl ? rowIndexEl.value.trim() : '';

        const payload = { changes };
        if (rowIndexText !== '') {
            payload.row_index = Number(rowIndexText);
        }

        const resultsEl = document.getElementById('whatIfResults');
        if (resultsEl) {
            resultsEl.style.display = 'block';
            resultsEl.innerHTML = '<div class="recommendation-text">Running simulation...</div>';
        }

        const response = await fetch(`${API_URL}/simulate-what-if`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        if (!result.success) {
            if (resultsEl) {
                let details = '';
                if (result.validation) {
                    const entries = Object.entries(result.validation);
                    details = entries.map(([feature, info]) => {
                        const allowed = (info.allowed_values || []).slice(0, 8).join(', ');
                        return `<div class="recommendation-text"><strong>${escapeHTML(feature)}:</strong> ${escapeHTML(info.message || 'Invalid value.')} ${allowed ? `Allowed: ${escapeHTML(allowed)}` : ''}</div>`;
                    }).join('');
                }
                resultsEl.innerHTML = `<div class="error-section">${escapeHTML(result.error || 'Simulation failed.')}</div>${details}`;
            }
            return;
        }

        renderWhatIfResult(result);
    } catch (error) {
        const resultsEl = document.getElementById('whatIfResults');
        if (resultsEl) {
            resultsEl.style.display = 'block';
            resultsEl.innerHTML = `<div class="error-section">Error running simulation: ${error.message}</div>`;
        }
    }
}

function formatPrediction(pred) {
    if (!pred) return 'N/A';
    if (typeof pred.score === 'number') {
        return `${pred.score.toFixed(4)} risk score`;
    }
    if (pred.raw_prediction !== undefined) {
        return String(pred.raw_prediction);
    }
    return 'N/A';
}

function renderWhatIfResult(result) {
    const resultsEl = document.getElementById('whatIfResults');
    if (!resultsEl) return;

    const oldPrediction = result.old_prediction || {};
    const newPrediction = result.new_prediction || {};
    const comparison = result.comparison || {};

    let comparisonHtml = '';
    if (comparison.comparison_available) {
        const delta = Number(comparison.delta || 0);
        const sign = delta >= 0 ? '+' : '';
        comparisonHtml = `
            <div class="recommendation-text"><strong>Old prediction:</strong> ${formatPrediction(oldPrediction)}</div>
            <div class="recommendation-text"><strong>New prediction:</strong> ${formatPrediction(newPrediction)}</div>
            <div class="recommendation-text"><strong>Change:</strong> ${sign}${delta.toFixed(4)} (${comparison.direction || 'changed'})</div>
        `;
    } else {
        comparisonHtml = `
            <div class="recommendation-text"><strong>Old prediction:</strong> ${formatPrediction(oldPrediction)}</div>
            <div class="recommendation-text"><strong>New prediction:</strong> ${formatPrediction(newPrediction)}</div>
            <div class="recommendation-text">${comparison.summary || 'Comparison summary unavailable.'}</div>
        `;
    }

    resultsEl.style.display = 'block';
    resultsEl.innerHTML = `
        <h4>Simulation Result</h4>
        ${comparisonHtml}
        <div style="margin-top:10px;"><strong>Insight:</strong>${formatRichTextHTML(result.insight || 'No explanation generated.', 'recommendation-text')}</div>
    `;
}
