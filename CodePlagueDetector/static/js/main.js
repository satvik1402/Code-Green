// Main JavaScript for DSA Plagiarism Detection System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize CodeMirror editor
    initializeCodeEditor();
    
    // Setup problem selection handler
    setupProblemSelection();
    
    // Setup form validation
    setupFormValidation();
    
    // Add loading states
    setupLoadingStates();
});

let codeEditor = null;

// Fast typing detection variables
let typingDetectionState = {
    charCount: 0,
    lastTypingTime: 0,
    typingWarningShown: false,
    // Configure sensitivity
    rapidThreshold: 30, // Characters per second threshold
    suddenIncreaseThreshold: 50, // Sudden character count increase
    fastTypingBlocked: false,
    lastContent: ''
};

function initializeCodeEditor() {
    const textarea = document.getElementById('codeEditor');
    if (!textarea) return;
    
    // Create a hidden input for form submission
    const fastTypingField = document.createElement('input');
    fastTypingField.type = 'hidden';
    fastTypingField.name = 'fast_typing_detected';
    fastTypingField.value = 'false';
    fastTypingField.id = 'fastTypingField';
    textarea.parentNode.appendChild(fastTypingField);
    
    codeEditor = CodeMirror.fromTextArea(textarea, {
        mode: 'python',
        theme: 'material-darker',
        lineNumbers: true,
        indentUnit: 4,
        indentWithTabs: false,
        lineWrapping: true,
        matchBrackets: true,
        autoCloseBrackets: true,
        foldGutter: true,
        gutters: ['CodeMirror-linenumbers', 'CodeMirror-foldgutter'],
        extraKeys: {
            'Tab': function(cm) {
                if (cm.somethingSelected()) {
                    cm.indentSelection('add');
                } else {
                    cm.replaceSelection(Array(cm.getOption('indentUnit') + 1).join(' '));
                }
            },
            'Shift-Tab': function(cm) {
                cm.indentSelection('subtract');
            },
            'Ctrl-/': function(cm) {
                cm.toggleComment();
            }
        },
        placeholder: 'Enter your solution here...\n\n# Example:\ndef solution():\n    # Your code here\n    pass'
    });
    
    // Set initial size
    codeEditor.setSize('100%', '400px');
    
    // Add custom styling
    codeEditor.getWrapperElement().classList.add('border', 'rounded');
    
    // Auto-refresh when visible
    setTimeout(() => {
        codeEditor.refresh();
    }, 100);
    
    // Add fast typing detection
    setupFastTypingDetection(codeEditor);
}

function setupProblemSelection() {
    const problemSelect = document.getElementById('problemSelect');
    const problemDescription = document.getElementById('problemDescription');
    const descriptionText = document.getElementById('descriptionText');
    const functionSignature = document.getElementById('functionSignature');
    
    if (!problemSelect) return;
    
    problemSelect.addEventListener('change', function() {
        const selectedOption = this.options[this.selectedIndex];
        
        if (selectedOption.value) {
            // Show problem description
            problemDescription.style.display = 'block';
            
            // Update description content
            const description = selectedOption.getAttribute('data-description');
            const signature = selectedOption.getAttribute('data-signature');
            
            descriptionText.innerHTML = formatDescription(description);
            functionSignature.textContent = signature;
            
            // Update code editor with function signature
            if (codeEditor && signature) {
                const currentCode = codeEditor.getValue().trim();
                if (!currentCode || currentCode === codeEditor.getOption('placeholder')) {
                    codeEditor.setValue(signature + '\n    # Your implementation here\n    pass');
                    codeEditor.setCursor(codeEditor.lineCount() - 1, 0);
                }
            }
            
            // Add animation
            problemDescription.classList.add('fade-in');
        } else {
            problemDescription.style.display = 'none';
        }
    });
}

function formatDescription(description) {
    if (!description) return '';
    
    // Convert newlines to HTML breaks and add basic formatting
    return description
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>')
        .replace(/Example:/g, '<strong>Example:</strong>')
        .replace(/Input:/g, '<strong>Input:</strong>')
        .replace(/Output:/g, '<strong>Output:</strong>')
        .replace(/Explanation:/g, '<strong>Explanation:</strong>')
        .replace(/Note:/g, '<strong>Note:</strong>');
}

function setupFormValidation() {
    const form = document.getElementById('submissionForm');
    if (!form) return;
    
    form.addEventListener('submit', function(e) {
        console.log('Form submission started');
        
        const problemId = document.getElementById('problemSelect').value;
        const code = codeEditor ? codeEditor.getValue().trim() : '';
        
        console.log('Problem ID:', problemId);
        console.log('Code length:', code.length);
        
        // Validate problem selection
        if (!problemId) {
            e.preventDefault();
            showAlert('Please select a problem to solve.', 'warning');
            return false;
        }
        
        // Validate code input
        if (!code || code === 'pass' || code.length < 10) {
            e.preventDefault();
            showAlert('Please enter a meaningful solution before submitting.', 'warning');
            codeEditor?.focus();
            return false;
        }
        
        // Update textarea with editor content before submission
        if (codeEditor) {
            const textarea = document.querySelector('textarea[name="code"]');
            if (textarea) {
                textarea.value = code;
                console.log('Updated textarea with code');
            }
        }
        
        // Show loading state
        showLoadingState(true);
        
        console.log('Form validation passed, submitting...');
        return true;
    });
}

function isValidPythonCode(code) {
    // Basic validation - check for common Python syntax patterns
    const lines = code.split('\n').filter(line => line.trim());
    
    // Must have at least one non-empty line
    if (lines.length === 0) return false;
    
    // Check for basic function definition or class
    const hasFunction = /def\s+\w+\s*\(/.test(code);
    const hasClass = /class\s+\w+/.test(code);
    const hasBasicCode = /[=+\-*\/]/.test(code) || /\bif\b|\bfor\b|\bwhile\b/.test(code);
    
    return hasFunction || hasClass || hasBasicCode;
}

function setupLoadingStates() {
    // Add loading spinner to submit button on form submission
    const submitBtn = document.querySelector('button[type="submit"]');
    if (submitBtn) {
        const originalText = submitBtn.innerHTML;
        
        window.showLoadingState = function(show) {
            if (show) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = `
                    <span class="spinner-border spinner-border-sm me-2" role="status"></span>
                    Analyzing Code...
                `;
            } else {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalText;
            }
        };
    }
}

function setupFastTypingDetection(editor) {
    if (!editor) return;
    
    const initialTime = Date.now();
    typingDetectionState.lastTypingTime = initialTime;
    
    // Track typing speed
    editor.on('changes', function(cm, changes) {
        const currentTime = Date.now();
        const timeDiff = (currentTime - typingDetectionState.lastTypingTime) / 1000; // in seconds
        typingDetectionState.lastTypingTime = currentTime;
        
        // Calculate content before and after changes
        const newContent = cm.getValue();
        const prevContent = typingDetectionState.lastContent;
        typingDetectionState.lastContent = newContent;
        
        // Skip first event
        if (prevContent === '') return;
        
        // Get the length difference
        const charDiff = Math.abs(newContent.length - prevContent.length);
        
        // Check for rapid typing or pasting
        if (typingDetectionState.fastTypingBlocked) {
            // If already blocked, prevent further typing
            if (prevContent !== newContent) {
                cm.setValue(prevContent);
                showAlert('Further typing blocked due to suspected cheating. Please contact the administrator.', 'danger');
            }
            return;
        }
        
        const isSuddenIncrease = charDiff > typingDetectionState.suddenIncreaseThreshold;
        const typingSpeed = timeDiff > 0 ? charDiff / timeDiff : 0;
        const isRapidTyping = typingSpeed > typingDetectionState.rapidThreshold;
        
        // Debug
        console.log('Typing speed:', typingSpeed, 'chars/sec, char diff:', charDiff);
        
        // Detect fast typing or pasting
        if ((isSuddenIncrease || isRapidTyping) && !typingDetectionState.typingWarningShown) {
            // First warning
            showAlert(
                'Fast typing or code pasting detected! This may be flagged as a potential cheating attempt. ' +
                'Another occurrence will block further typing.', 
                'warning'
            );
            
            // Update state
            typingDetectionState.typingWarningShown = true;
            
            // Update hidden form field
            const fastTypingField = document.getElementById('fastTypingField');
            if (fastTypingField) {
                fastTypingField.value = 'true';
            }
            
            return;
        }
        
        // Second occurrence
        if ((isSuddenIncrease || isRapidTyping) && typingDetectionState.typingWarningShown) {
            // Block further typing
            typingDetectionState.fastTypingBlocked = true;
            
            // Show error
            showAlert('Typing blocked due to suspected code pasting. Please contact the administrator.', 'danger');
            
            // Prevent form submission
            const form = document.getElementById('submissionForm');
            if (form) {
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.disabled = true;
                }
            }
        }
    });
}

function showAlert(message, type = 'info') {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of container
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Utility functions for code editor
function insertCodeTemplate(template) {
    if (codeEditor) {
        const cursor = codeEditor.getCursor();
        codeEditor.replaceRange(template, cursor);
        codeEditor.focus();
    }
}

function formatCode() {
    if (codeEditor) {
        const code = codeEditor.getValue();
        // Basic Python formatting (add proper indentation)
        const lines = code.split('\n');
        let indentLevel = 0;
        const formattedLines = [];
        
        lines.forEach(line => {
            const trimmed = line.trim();
            if (!trimmed) {
                formattedLines.push('');
                return;
            }
            
            // Decrease indent for certain keywords
            if (trimmed.startsWith('except') || trimmed.startsWith('finally') || 
                trimmed.startsWith('elif') || trimmed.startsWith('else')) {
                indentLevel = Math.max(0, indentLevel - 1);
            }
            
            formattedLines.push('    '.repeat(indentLevel) + trimmed);
            
            // Increase indent after certain keywords
            if (trimmed.endsWith(':')) {
                indentLevel++;
            }
        });
        
        codeEditor.setValue(formattedLines.join('\n'));
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+Enter to submit form
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        const form = document.getElementById('submissionForm');
        if (form) {
            form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
        }
    }
    
    // Ctrl+Shift+F to format code
    if (e.ctrlKey && e.shiftKey && e.key === 'F') {
        e.preventDefault();
        formatCode();
    }
});

// Handle page visibility change to refresh CodeMirror
document.addEventListener('visibilitychange', function() {
    if (!document.hidden && codeEditor) {
        setTimeout(() => codeEditor.refresh(), 100);
    }
});

// Window resize handler
window.addEventListener('resize', function() {
    if (codeEditor) {
        setTimeout(() => codeEditor.refresh(), 100);
    }
});

// Export functions for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        formatDescription,
        isValidPythonCode,
        showAlert
    };
}
