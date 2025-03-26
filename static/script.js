document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const dropArea = document.getElementById('drop-area');
    const fileUpload = document.getElementById('file-upload');
    const fileInfo = document.getElementById('file-info');
    const uploadContainer = document.getElementById('upload-container');
    const comparisonContainer = document.getElementById('comparison-container');
    const originalImage = document.getElementById('original-image');
    const colorizedImage = document.getElementById('colorized-image');
    const loadingOverlay = document.getElementById('loading-overlay');
    const newImageBtn = document.getElementById('new-image-btn');
    const downloadBtn = document.getElementById('download-btn');
    const shareBtn = document.getElementById('share-btn');
    const shareModal = document.getElementById('share-modal');
    const closeModal = document.querySelector('.close');
    const shareLinkInput = document.getElementById('share-link-input');
    const copyLinkBtn = document.getElementById('copy-link-btn');

    // API endpoint
    const API_URL = window.location.origin;
    
    // Current uploaded image data
    let currentUploadedFile = null;
    let colorizedImageUrl = null;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);

    // Handle file input change
    fileUpload.addEventListener('change', handleFiles, false);

    // Handle click on dropArea
    dropArea.addEventListener('click', function() {
        fileUpload.click();
    });

    // New image button
    newImageBtn.addEventListener('click', function() {
        switchToUploadView();
    });

    // Download button
    downloadBtn.addEventListener('click', function() {
        if (colorizedImageUrl) {
            downloadImage(colorizedImageUrl);
        }
    });

    // Share button
    shareBtn.addEventListener('click', function() {
        if (colorizedImageUrl) {
            showShareModal();
        }
    });

    // Close modal
    closeModal.addEventListener('click', function() {
        shareModal.style.display = 'none';
    });

    // Copy link button
    copyLinkBtn.addEventListener('click', function() {
        shareLinkInput.select();
        document.execCommand('copy');
        copyLinkBtn.textContent = 'Copied!';
        setTimeout(() => {
            copyLinkBtn.textContent = 'Copy';
        }, 2000);
    });

    // Close modal when clicking outside of it
    window.addEventListener('click', function(event) {
        if (event.target === shareModal) {
            shareModal.style.display = 'none';
        }
    });

    // Helper functions
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropArea.classList.add('dragover');
    }

    function unhighlight() {
        dropArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles({ target: { files: files } });
    }

    function handleFiles(e) {
        const files = e.target.files;
        if (files.length) {
            currentUploadedFile = files[0];
            fileInfo.textContent = files[0].name;
            
            // Validate file type
            const fileType = files[0].type;
            if (!['image/jpeg', 'image/jpg', 'image/png', 'image/gif'].includes(fileType)) {
                showError('Please upload a valid image file (JPEG, PNG, GIF)');
                return;
            }
            
            // Upload the file
            uploadImage(files[0]);
        }
    }

    function uploadImage(file) {
        // Create FormData
        const formData = new FormData();
        formData.append('file', file);
        
        // Show loading state
        switchToComparisonView();
        loadingOverlay.style.display = 'flex';
        
        // Set original image preview
        const reader = new FileReader();
        reader.onload = function(e) {
            originalImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        // Send to server
        fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Display colorized image
                colorizedImageUrl = `${API_URL}/images/${data.colorized}`;
                colorizedImage.src = colorizedImageUrl;
                colorizedImage.onload = function() {
                    loadingOverlay.style.display = 'none';
                };
                
                // Update share link
                shareLinkInput.value = window.location.href;
                
                // Add processing time if needed
                console.log(`Processing time: ${data.processing_time.toFixed(2)}s`);
            } else {
                showError('Error processing image');
                switchToUploadView();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Error uploading image. Please try again.');
            switchToUploadView();
        });
    }

    function switchToComparisonView() {
        uploadContainer.style.display = 'none';
        comparisonContainer.style.display = 'block';
    }

    function switchToUploadView() {
        comparisonContainer.style.display = 'none';
        uploadContainer.style.display = 'block';
        // Reset file input
        fileUpload.value = '';
        fileInfo.textContent = 'No file selected';
        currentUploadedFile = null;
    }

    function showShareModal() {
        shareModal.style.display = 'flex';
        // Set share link
        shareLinkInput.value = window.location.href;
    }

    function downloadImage(imageUrl) {
        const link = document.createElement('a');
        link.href = imageUrl;
        link.download = 'colorized_image.jpg';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    function showError(message) {
        // You can implement a more sophisticated error notification system
        alert(message);
    }
    
    // Social media share buttons functionality
    document.querySelectorAll('.share-options .share-button').forEach(button => {
        button.addEventListener('click', function() {
            let url = encodeURIComponent(window.location.href);
            let text = encodeURIComponent('Check out this colorized image using ChromaVision AI!');
            let shareUrl;
            
            if (button.classList.contains('facebook')) {
                shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${url}`;
            } else if (button.classList.contains('twitter')) {
                shareUrl = `https://twitter.com/intent/tweet?url=${url}&text=${text}`;
            } else if (button.classList.contains('pinterest')) {
                shareUrl = `https://pinterest.com/pin/create/button/?url=${url}&description=${text}&media=${encodeURIComponent(colorizedImageUrl)}`;
            } else if (button.classList.contains('instagram')) {
                // Instagram doesn't have a direct share URL, show instructions instead
                alert('To share on Instagram, please save the image and upload it to your Instagram account.');
                return;
            }
            
            window.open(shareUrl, '_blank', 'width=600,height=400');
        });
    });

    // Handle colorized image load error
    colorizedImage.addEventListener('error', function() {
        loadingOverlay.style.display = 'none';
        showError('Error loading colorized image. Please try again.');
    });

    // Theme toggle functionality could be added here
    // This is a placeholder for future implementation
    function toggleDarkMode() {
        document.body.classList.toggle('dark-mode');
        // Save preference to localStorage
    }

    // Check if dark mode is preferred
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        // toggleDarkMode();
    }

    // Optional: Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Escape key closes modal
        if (e.key === 'Escape' && shareModal.style.display === 'flex') {
            shareModal.style.display = 'none';
        }
    });
});