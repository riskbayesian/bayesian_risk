window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
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
    	// Add listener to  event
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

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

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

    // Add event listeners for dropdown changes
    document.getElementById('environment-select').addEventListener('change', updateDemoContent);
    document.getElementById('image-type-select').addEventListener('change', updateDemoContent);
    
    // Don't initialize content on page load - wait for user selection

})
