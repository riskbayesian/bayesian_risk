window.HELP_IMPROVE_VIDEOJS = false;

// CACHE BUST: Updated 2025-10-19 14:12 - All interpolation code removed
console.log('=== JavaScript file loaded at 14:12 - NO INTERPOLATION CODE ===');

$(document).ready(function() {
    console.log('=== JavaScript loaded successfully ===');
    
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");
    });

    var options = {
        slidesToScroll: 1,
        slidesToShow: 3,
        loop: true,
        infinite: true,
        autoplay: false,
        autoplaySpeed: 3000,
    }

    // Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
        // Add listener to event
        carousels[i].on('before:show', state => {
            console.log(state);
        });
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
        // bulmaCarousel instance is available as element.bulmaCarousel
        element.bulmaCarousel.on('before-show', function(state) {
            console.log(state);
        });
    }

    bulmaSlider.attach();

    // Demonstrations section functionality
    function updateDemoContent() {
        const environment = document.getElementById('environment-select').value;
        const imageType = document.getElementById('image-type-select').value;
        const demoContent = document.getElementById('demo-content');
        
        // Hide content if either dropdown is set to "None"
        if (environment === '' || imageType === '') {
            demoContent.style.display = 'none';
            return;
        }
        
        // Show the content area when both selections are made
        demoContent.style.display = 'flex';
        
        // Convert image type to filename format (lowercase, replace spaces with underscores)
        const imageTypeFilename = imageType.toLowerCase().replace(/\s+/g, '_');
        
        // Use the environment name directly as the folder path
        const folderPath = environment;
        
        // Update video source
        const videoElement = document.getElementById('demo-video');
        const videoSource = document.getElementById('video-source');
        // Use H.264 versions for all videos
        const videoUrl = `./static/scenes/${folderPath}/videos/${imageTypeFilename}_h264.mp4`;
        
        // Clear existing sources
        videoElement.innerHTML = '';
        
        // Create new source element
        const source = document.createElement('source');
        source.src = videoUrl;
        source.type = 'video/mp4';
        
        // Add source to video element
        videoElement.appendChild(source);
        videoElement.appendChild(document.createTextNode('Your browser does not support the video tag.'));
        
        // Add error handling
        videoElement.onerror = function(e) {
            console.error('Video load error:', e);
            console.error('Failed to load video:', videoUrl);
            console.error('Video element error details:', videoElement.error);
        };
        
        videoElement.onloadeddata = function() {
            console.log('Video loaded successfully:', videoUrl);
        };
        
        videoElement.oncanplay = function() {
            console.log('Video can start playing:', videoUrl);
        };
        
        // Load the video
        videoElement.load();
        
        // Update risk graph image
        const riskGraph = document.getElementById('risk-graph');
        riskGraph.src = `./static/scenes/${folderPath}/risk_graphs/${imageTypeFilename}.png`;
        riskGraph.alt = `${imageType} Risk Graph for Environment ${environment}`;
    }

    // Hardware Experiments functionality
    function updateHardwareContent() {
        console.log('=== updateHardwareContent called ===');
        const trialName = document.getElementById('hardware-video-select').value;
        const hardwareContent = document.getElementById('hardware-content');
        
        console.log('Trial name:', trialName);
        
        // Hide content if no trial is selected
        if (trialName === '') {
            hardwareContent.style.display = 'none';
            return;
        }
        
        // Show the content area when a trial is selected
        hardwareContent.style.display = 'flex';
        
        // Define the 4 video types and their corresponding elements
        const videoTypes = ['ours', 'vla', 'human', 'diffusion'];
        
        // Load each video
        videoTypes.forEach(videoType => {
            const videoElement = document.getElementById(`hardware-video-${videoType}`);
            const videoUrl = `./static/hardware_vids/${trialName}/${videoType}.mp4`;
            
            // Clear existing sources
            videoElement.innerHTML = '';
            
            // Create new source element
            const source = document.createElement('source');
            source.src = videoUrl;
            source.type = 'video/mp4';
            
            // Add source to video element
            videoElement.appendChild(source);
            videoElement.appendChild(document.createTextNode('Your browser does not support the video tag.'));
            
            // Add error handling
            videoElement.onerror = function(e) {
                console.error(`Hardware video load error (${videoType}):`, e);
                console.error('Failed to load video:', videoUrl);
            };
            
            videoElement.onloadeddata = function() {
                console.log(`Hardware video loaded successfully (${videoType}):`, videoUrl);
            };
            
            // Load the video
            videoElement.load();
        });
    }

    // Function to populate hardware video dropdown
    function populateHardwareDropdown() {
        console.log('=== populateHardwareDropdown called ===');
        
        // List of available hardware trials (each trial contains 4 videos: ours, vla, human, diffusion)
        const hardwareTrials = [
            'Trial_1',
            'Trial_2',
            'Trial_3',
            'Trial_4',
            'Trial_5',
            // Add more trials here as you create them, for example:
            // 'Trial_7',
            // 'Trial_8',
            // 'Experiment_1'
        ];
        
        console.log('Hardware trials:', hardwareTrials);
        
        // Populate hardware video dropdown
        const hardwareSelect = document.getElementById('hardware-video-select');
        if (!hardwareSelect) {
            console.error('Hardware select element not found!');
            return;
        }
        
        console.log('Found hardware select element:', hardwareSelect);
        
        hardwareSelect.innerHTML = '<option value="">None</option>';
        hardwareTrials.forEach(trial => {
            const option = document.createElement('option');
            option.value = trial;
            option.textContent = trial.replace(/_/g, ' '); // Clean up display name
            hardwareSelect.appendChild(option);
            console.log('Added option:', trial);
        });
        
        console.log('Hardware dropdown populated successfully');
    }

    // Add event listeners for dropdown changes
    document.getElementById('environment-select').addEventListener('change', updateDemoContent);
    document.getElementById('image-type-select').addEventListener('change', updateDemoContent);
    document.getElementById('hardware-video-select').addEventListener('change', updateHardwareContent);
    
    // Populate hardware dropdown on page load
    populateHardwareDropdown();
    
    // Don't initialize content on page load - wait for user selection

})