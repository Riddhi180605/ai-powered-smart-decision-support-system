(function () {
    // API Configuration - Dynamic for both development and production
    function getAPIUrl() {
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://localhost:8001';
        }
        return window.location.origin;
    }
    const CHATBOT_API_URL = getAPIUrl();
    const CHAT_HISTORY_KEY = 'chatbotConversationHistory';

    function escapeHtml(text) {
        return String(text || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }

    function parseMarkdown(text) {
        if (!text) return '';
        let html = escapeHtml(text);

        // Split by double newlines to preserve paragraph structure
        const paragraphs = html.split(/\n\n+/);

        const processedParagraphs = paragraphs.map(para => {
            let processed = para;

            // Check if this paragraph contains list items
            const isListParagraph = /^\s*[-•]\s+|^\s*\d+\.\s+/.test(processed);

            if (isListParagraph) {
                // Process as list
                const lines = processed.split('\n');
                let listHtml = '<ul class="cbot-list">';

                lines.forEach(line => {
                    // Match bullet points or numbered lists
                    const bulletMatch = line.match(/^([-•]\s+)(.+)$/);
                    const numberedMatch = line.match(/^(\d+\.\s+)(.+)$/);

                    if (bulletMatch) {
                        let content = bulletMatch[2];
                        // Apply bold formatting to this line
                        content = content.replace(/\*\*([^\*]+)\*\*/g, '<strong class="cbot-bold">$1</strong>');
                        listHtml += `<li class="cbot-list-item">• ${content}</li>`;
                    } else if (numberedMatch) {
                        let content = numberedMatch[2];
                        // Apply bold formatting to this line
                        content = content.replace(/\*\*([^\*]+)\*\*/g, '<strong class="cbot-bold">$1</strong>');
                        listHtml += `<li class="cbot-list-item">${numberedMatch[1]}${content}</li>`;
                    } else if (line.trim()) {
                        // Non-list line in list context
                        let content = line.replace(/\*\*([^\*]+)\*\*/g, '<strong class="cbot-bold">$1</strong>');
                        listHtml += `<li class="cbot-list-item">${content}</li>`;
                    }
                });
                listHtml += '</ul>';
                return listHtml;
            } else {
                // Process as regular paragraph
                // Convert headers (# Title, ## Title, ### Title)
                if (processed.match(/^#+\s+/)) {
                    const headerMatch = processed.match(/^(#+)\s+(.+)$/);
                    if (headerMatch) {
                        const level = Math.min(headerMatch[1].length + 1, 4);
                        let content = headerMatch[2];
                        content = content.replace(/\*\*([^\*]+)\*\*/g, '<strong class="cbot-bold">$1</strong>');
                        return `<h${level} class="cbot-heading">${content}</h${level}>`;
                    }
                }

                // Convert bold text **text** (must be before italic)
                processed = processed.replace(/\*\*([^\*]+)\*\*/g, '<strong class="cbot-bold">$1</strong>');

                // Convert italic text *text*
                processed = processed.replace(/\*([^\*]+)\*/g, '<em>$1</em>');

                // Preserve single line breaks inside a paragraph.
                processed = processed.replace(/\n/g, '<br>');

                // Wrap in paragraph
                if (processed.trim()) {
                    return `<p class="cbot-paragraph">${processed}</p>`;
                }
                return '';
            }
        });

        return processedParagraphs.filter(p => p).join('');
    }

    function createMessage(role, text) {
        const item = document.createElement('div');
        item.className = `cbot-msg ${role}`;
        item.innerHTML = parseMarkdown(text);
        // Allow safe HTML content for bot messages
        if (role === 'bot' || role === 'system') {
            item.classList.add('cbot-formatted');
        }
        return item;
    }

    function getChatHistory() {
        try {
            const raw = sessionStorage.getItem(CHAT_HISTORY_KEY);
            const parsed = raw ? JSON.parse(raw) : [];
            return Array.isArray(parsed) ? parsed : [];
        } catch (error) {
            return [];
        }
    }

    function saveChatHistory(history) {
        sessionStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(history));
    }

    function appendMessage(container, role, text, options = {}) {
        const { persist = true } = options;
        container.appendChild(createMessage(role, text));
        container.scrollTop = container.scrollHeight;

        if (persist) {
            const history = getChatHistory();
            history.push({ role, text });
            saveChatHistory(history);
        }
    }

    function renderStoredMessages(container) {
        container.innerHTML = '';
        const history = getChatHistory();
        history.forEach(msg => {
            appendMessage(container, msg.role || 'system', msg.text || '', { persist: false });
        });
    }

    async function getProjectState() {
        try {
            const response = await fetch(`${CHATBOT_API_URL}/project-state`);
            const result = await response.json();
            return result.project_state || {};
        } catch (error) {
            return {};
        }
    }

    async function ensureIndex(messagesContainer) {
        try {
            const response = await fetch(`${CHATBOT_API_URL}/chatbot/build-index`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const result = await response.json();
            if (!response.ok || !result.success) {
                throw new Error(result.detail || result.error || 'Unable to build chatbot index.');
            }
            return true;
        } catch (error) {
            appendMessage(messagesContainer, 'system', `Index build failed: ${error.message}`);
            return false;
        }
    }

    async function askQuestion(query, messagesContainer) {
        const response = await fetch(`${CHATBOT_API_URL}/chatbot/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        const result = await response.json();
        if (!response.ok || !result.success) {
            const msg = result.detail || result.error || result.answer || 'Chatbot could not answer right now.';
            throw new Error(msg);
        }

        return result.answer || 'No answer found.';
    }

    function buildWidget() {
        const launcher = document.createElement('button');
        launcher.className = 'cbot-launcher';
        launcher.type = 'button';
        launcher.innerHTML = [
            '<span class="cbot-launch-icon">💬</span>',
            '<span class="cbot-launch-text">Ask AI Assistant...</span>'
        ].join('');

        const backdrop = document.createElement('div');
        backdrop.className = 'cbot-backdrop';

        const panel = document.createElement('aside');
        panel.className = 'cbot-panel';
        panel.innerHTML = `
            <div class="cbot-header">
                <h3 class="cbot-title">AI Assistant</h3>
                <div class="cbot-header-controls">
                    <button class="cbot-fullscreen" type="button" aria-label="Open chatbot in fullscreen" title="Fullscreen">⤢</button>
                    <button class="cbot-close" type="button" aria-label="Close chatbot">✕</button>
                </div>
            </div>
            <div class="cbot-messages" id="cbotMessages"></div>
            <div class="cbot-input-wrap">
                <input id="cbotInput" type="text" placeholder="Ask anything about your data..." />
                <button id="cbotSend" type="button">Send</button>
            </div>
        `;

        document.body.appendChild(launcher);
        document.body.appendChild(backdrop);
        document.body.appendChild(panel);

        const closeBtn = panel.querySelector('.cbot-close');
        const fullscreenBtn = panel.querySelector('.cbot-fullscreen');
        const messages = panel.querySelector('#cbotMessages');
        const input = panel.querySelector('#cbotInput');
        const sendBtn = panel.querySelector('#cbotSend');
        let isFullscreen = false;

        const updateFullscreenUi = () => {
            panel.classList.toggle('fullscreen', isFullscreen);
            if (isFullscreen) {
                fullscreenBtn.setAttribute('aria-label', 'Exit fullscreen mode');
                fullscreenBtn.setAttribute('title', 'Exit fullscreen');
                fullscreenBtn.textContent = '🗗';
            } else {
                fullscreenBtn.setAttribute('aria-label', 'Open chatbot in fullscreen');
                fullscreenBtn.setAttribute('title', 'Fullscreen');
                fullscreenBtn.textContent = '⤢';
            }
        };

        const openPanel = async () => {
            launcher.classList.add('hidden');
            backdrop.classList.add('open');
            panel.classList.add('open');
            setTimeout(() => input.focus(), 200);

            // Load and display project context
            await displayProjectContext(messages);
        };

        const closePanel = () => {
            panel.classList.remove('open');
            backdrop.classList.remove('open');
            launcher.classList.remove('hidden');
            isFullscreen = false;
            updateFullscreenUi();
        };

        const toggleFullscreen = () => {
            isFullscreen = !isFullscreen;
            updateFullscreenUi();
        };

        launcher.addEventListener('click', openPanel);
        closeBtn.addEventListener('click', closePanel);
        fullscreenBtn.addEventListener('click', toggleFullscreen);
        backdrop.addEventListener('click', closePanel);

        let indexReady = false;
        let busy = false;

        async function displayProjectContext(messagesContainer) {
            const history = getChatHistory();

            if (history.length === 0) {
                appendMessage(messagesContainer, 'system', 'Hello , i am your chat buddy.');
                return;
            }

            renderStoredMessages(messagesContainer);
        }

        function getExampleQuestionsForStage(stage) {
            const questions = {
                'initial': ['Upload a dataset first!'],
                'data_loaded': ['What\'s in my dataset?', 'Show column names', 'Any missing values?'],
                'data_cleaned': ['What changed during cleaning?', 'Dataset summary', 'Ready to train?'],
                'model_trained': ['Which features matter most?', 'Model performance?', 'Next steps?', 'How to improve?'],
                'ai_insights_generated': ['Key factors?', 'Business recommendations', 'What if I change this?', 'Risk analysis?', 'Trend insights?']
            };
            return questions[stage] || ['Tell me about my project'];
        }

        const onSend = async () => {
            if (busy) {
                return;
            }

            const query = input.value.trim();
            if (!query) {
                return;
            }

            appendMessage(messages, 'user', query);
            input.value = '';
            busy = true;
            sendBtn.disabled = true;

            // Show typing indicator
            const typingMessage = createMessage('bot', '🔄 Thinking...');
            messages.appendChild(typingMessage);
            messages.scrollTop = messages.scrollHeight;

            const completeTyping = (finalText) => {
                if (typingMessage.parentNode) {
                    typingMessage.parentNode.removeChild(typingMessage);
                }
                appendMessage(messages, 'bot', finalText);
            };

            try {
                // First, try intelligent question answering endpoint
                const answerResponse = await fetch(`${CHATBOT_API_URL}/chatbot/answer-question`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: query })
                });

                const answerData = await answerResponse.json();

                if (answerData.success && answerData.answer) {
                    // Direct answer from question handler
                    completeTyping(answerData.answer);
                } else if (answerData.fallback_to_rag) {
                    // Should fall back to RAG
                    const state = await getProjectState();
                    if (state.chatbot_ready) {
                        // Build index if needed
                        if (!indexReady) {
                            typingMessage.textContent = 'Preparing knowledge base...';
                            indexReady = await ensureIndex(messages);
                        }

                        if (indexReady) {
                            // Use RAG chatbot
                            try {
                                const ragAnswer = await askQuestion(query, messages);
                                completeTyping(ragAnswer);
                            } catch (ragError) {
                                completeTyping(`I couldn't find an answer: ${ragError.message || 'Unknown error'}`);
                            }
                        }
                    } else {
                        completeTyping(answerData.message || "I need more context. Try asking about your dataset, cleaning, or model training.");
                    }
                } else {
                    completeTyping("I need more context. Try asking about your dataset, cleaning, or model training.");
                }
            } catch (error) {
                completeTyping(`Connection error: ${error.message || error}. Make sure the backend is running on localhost:8001`);
                console.error('Chat error:', error);
            } finally {
                busy = false;
                sendBtn.disabled = false;
            }
        };

        function getStagedHelpMessage(query, stage) {
            const queryLower = query.toLowerCase();

            if (stage === 'initial') {
                return '📋 Please upload a dataset first to get started. Go to the home page and upload a CSV or Excel file.';
            }

            if (stage === 'data_loaded') {
                if (queryLower.includes('clean') || queryLower.includes('format')) {
                    return '🧹 Click the "Clean Data" button to start data cleaning. I can help analyze issues after that!';
                }
                if (queryLower.includes('missing') || queryLower.includes('duplicate') || queryLower.includes('null')) {
                    return '📊 Let\'s clean your data to handle missing values and duplicates. This will prepare it for model training.';
                }
                return '📊 Your dataset is loaded! Next step: Clean your data by removing duplicates, handling missing values, etc.';
            }

            if (stage === 'data_cleaned') {
                if (queryLower.includes('train') || queryLower.includes('model') || queryLower.includes('ml')) {
                    return '🤖 Great! Your cleaned data is ready. Go to "ML Training" to train machine learning models on your dataset.';
                }
                return '✨ Data cleaned and ready! Next: Train ML models on this cleaned dataset for predictions and insights.';
            }

            if (stage === 'model_trained') {
                if (queryLower.includes('insight') || queryLower.includes('suggestion') || queryLower.includes('recommend')) {
                    return '🧠 Perfect! Click "Show AI Suggestions" to generate deep insights, business recommendations, and analysis from your trained model.';
                }
                if (queryLower.includes('what if') || queryLower.includes('scenario') || queryLower.includes('change')) {
                    return '🧪 Check the "What-If Simulation" tab to test scenarios and see how changes affect predictions!';
                }
                return '🤖 Model trained successfully! Next: Generate AI insights to understand patterns and get business recommendations.';
            }

            return '💡 I\'m here to help! Ask me about your analytics project at any stage.';
        }

        sendBtn.addEventListener('click', onSend);
        input.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                onSend();
            }
            if (event.key === 'Escape') {
                if (isFullscreen) {
                    isFullscreen = false;
                    updateFullscreenUi();
                    return;
                }
                closePanel();
            }
        });
    }

    window.addEventListener('DOMContentLoaded', buildWidget);
})();
