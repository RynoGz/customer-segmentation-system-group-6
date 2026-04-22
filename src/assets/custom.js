document.addEventListener('DOMContentLoaded', function() {
    console.log("Advanced UI Scripts Loaded.");

    // Feature 1: Smooth JS Page Fade-In
    document.body.style.opacity = 0;
    let opacity = 0;
    let timer = setInterval(function() {
        if (opacity >= 1) {
            clearInterval(timer);
        }
        document.body.style.opacity = opacity;
        opacity += 0.05;
    }, 30); // Runs every 30ms

    // Feature 2: Dynamic Button State (DOM Manipulation)
    // Wait a brief moment to ensure Dash has rendered the button
    setTimeout(function() {
        document.addEventListener('click', function(e) {
            if (e.target && e.target.id === 'predict-btn') {
                let btn = e.target;
                let originalText = btn.innerText;
                
                // Change text to show activity
                btn.innerText = "Processing Data...";
                
                // Change it back after the Dash callback finishes (approx 1.5 seconds)
                setTimeout(function() {
                    btn.innerText = "GENERATE SEGMENT";
                }, 1500);
            }
        });
    }, 1000);
});