(function () {
    function markPageReady() {
        requestAnimationFrame(() => {
            document.body.classList.add('page-ready');
        });
    }

    function applyStagger(root) {
        const scope = root || document;
        const groups = scope.querySelectorAll('.stats-grid, .cleaning-summary, .results-summary, .action-buttons, .dataset-summary, .tabs');

        groups.forEach(group => {
            const items = group.children;
            Array.from(items).forEach((item, index) => {
                item.classList.add('stagger-item');
                item.style.setProperty('--stagger-delay', `${index * 80}ms`);
            });
        });
    }

    function revealElements(root) {
        const scope = root || document;
        const selectors = [
            '.section',
            '.table-responsive',
            '.info-table',
            '.results-table',
            '.model-compare-table',
            '.summary-card',
            '.suggestion-card',
            '.comparison-box',
            '.upload-section',
            '.stat-card',
            '.option-group',
            '.result-item',
            '.ai-suggestions-container'
        ];

        const nodes = scope.querySelectorAll(selectors.join(','));
        nodes.forEach(node => {
            if (!node.classList.contains('a-fade-up') && !node.classList.contains('stagger-item')) {
                node.classList.add('a-fade-up');
            }
        });

        const revealTargets = scope.querySelectorAll('.a-fade-up, .stagger-item');
        revealTargets.forEach(node => {
            setTimeout(() => node.classList.add('is-visible'), 16);
        });
    }

    function animateCharts(root) {
        const scope = root || document;
        const chartNodes = scope.querySelectorAll('canvas, svg, .chart, .plot, .graph');
        chartNodes.forEach(node => {
            node.classList.add('chart-animate');
            requestAnimationFrame(() => node.classList.add('chart-visible'));
        });
    }

    function setupRipple() {
        const interactive = document.querySelectorAll('.btn, .tab-button, .whatif-close-btn, .quick-presets button, .change-item button, .home-icon-btn, .cbot-input-wrap button, .cbot-close, .cbot-fullscreen');

        interactive.forEach(button => {
            if (button.dataset.rippleBound === '1') {
                return;
            }
            button.dataset.rippleBound = '1';

            button.addEventListener('pointerdown', () => {
                button.classList.add('is-pressed');
            });
            button.addEventListener('pointerup', () => {
                button.classList.remove('is-pressed');
            });
            button.addEventListener('pointerleave', () => {
                button.classList.remove('is-pressed');
            });

            button.addEventListener('click', (event) => {
                const rect = button.getBoundingClientRect();
                const ripple = document.createElement('span');
                const size = Math.max(rect.width, rect.height);
                ripple.className = 'ripple';
                ripple.style.width = `${size}px`;
                ripple.style.height = `${size}px`;
                ripple.style.left = `${event.clientX - rect.left - size / 2}px`;
                ripple.style.top = `${event.clientY - rect.top - size / 2}px`;
                button.appendChild(ripple);

                ripple.addEventListener('animationend', () => {
                    ripple.remove();
                }, { once: true });
            });
        });
    }

    function setupInputStates() {
        const inputs = document.querySelectorAll('input, textarea, .form-control, .chat-input-row input, .whatif-grid input');

        inputs.forEach(input => {
            input.addEventListener('input', () => {
                if (input.value && input.value.trim() !== '') {
                    input.classList.add('is-typing');
                } else {
                    input.classList.remove('is-typing');
                }
            });
        });
    }

    function setActiveNavItem() {
        const page = window.location.pathname.split('/').pop() || 'index.html';
        const map = {
            'index.html': 'a[href="index.html"]',
            'dataset.html': '#cleanBtn',
            'cleaning.html': '#confirmCleaning',
            'cleaning-results.html': '#mlTrainBtn',
            'ml-training.html': '#trainBtn',
            'suggestions.html': '#whatIfNavBtn'
        };

        const nav = document.querySelector('.nav-buttons');
        if (!nav) {
            return;
        }

        nav.querySelectorAll('.btn, .tab-button, .home-icon-btn').forEach(item => {
            item.classList.remove('is-active');
        });

        const activeSelector = map[page];
        if (!activeSelector) {
            return;
        }

        const activeEl = nav.querySelector(activeSelector);
        if (activeEl) {
            activeEl.classList.add('is-active');
        }
    }

    function setupNavCollapse() {
        const nav = document.querySelector('.nav-buttons');
        if (!nav) {
            return;
        }

        const navItems = nav.querySelectorAll('.btn, .tab-button, .home-icon-btn');
        if (navItems.length < 3) {
            return;
        }

        if (document.querySelector('.nav-collapse-toggle')) {
            return;
        }

        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'btn btn-secondary nav-collapse-toggle';
        toggle.textContent = 'Toggle Menu';
        toggle.setAttribute('aria-expanded', 'true');

        let collapsed = false;
        toggle.addEventListener('click', () => {
            collapsed = !collapsed;
            nav.classList.toggle('is-collapsed', collapsed);
            toggle.textContent = collapsed ? 'Open Menu' : 'Hide Menu';
            toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
        });

        nav.parentNode.insertBefore(toggle, nav);
    }

    function setupAnchorNavigation() {
        const anchors = document.querySelectorAll('a[href^="#"]');

        anchors.forEach(anchor => {
            anchor.addEventListener('click', (event) => {
                const href = anchor.getAttribute('href');
                if (!href || href === '#') {
                    return;
                }

                const target = document.querySelector(href);
                if (!target) {
                    return;
                }

                event.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                history.pushState(null, '', href);
            });
        });
    }

    function bindTabAnimationSync() {
        document.addEventListener('click', (event) => {
            const target = event.target;
            if (!(target instanceof HTMLElement)) {
                return;
            }

            if (target.classList.contains('tab-button')) {
                document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('is-active'));
                target.classList.add('is-active');

                setTimeout(() => {
                    const activeTab = document.querySelector('.tab-content.active');
                    if (activeTab) {
                        applyStagger(activeTab);
                        revealElements(activeTab);
                    }
                }, 20);
            }
        });
    }

    function observeDynamicContent() {
        const pendingRoots = new Set();
        let rafId = 0;

        function flush() {
            rafId = 0;
            pendingRoots.forEach(root => {
                setupRipple();
                applyStagger(root);
                revealElements(root);
                animateCharts(root);
            });
            pendingRoots.clear();
        }

        const observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                if (mutation.addedNodes && mutation.addedNodes.length > 0) {
                    mutation.addedNodes.forEach(node => {
                        if (node instanceof HTMLElement) {
                            pendingRoots.add(node);
                        }
                    });
                }
            }

            if (pendingRoots.size && !rafId) {
                rafId = requestAnimationFrame(flush);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    function runDashboardReveal(root) {
        applyStagger(root || document);
        revealElements(root || document);
        animateCharts(root || document);
    }

    function bindCustomEvents() {
        document.addEventListener('dashboard:loaded', (event) => {
            runDashboardReveal(event.target instanceof HTMLElement ? event.target : document);
            setActiveNavItem();
        });

        document.addEventListener('charts:rendered', (event) => {
            animateCharts(event.target instanceof HTMLElement ? event.target : document);
        });
    }

    function init() {
        markPageReady();
        setupRipple();
        setupInputStates();
        setupNavCollapse();
        setupAnchorNavigation();
        bindTabAnimationSync();
        setActiveNavItem();
        runDashboardReveal(document);
        bindCustomEvents();
        observeDynamicContent();
    }

    document.addEventListener('DOMContentLoaded', init);
})();
