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
    }, 30);

    
    
    setTimeout(function() {
        document.addEventListener('click', function(e) {
            if (e.target && e.target.id === 'predict-btn') {
                let btn = e.target;
                let originalText = btn.innerText;
                
                
                btn.innerText = "Processing Data...";
                
                
                setTimeout(function() {
                    btn.innerText = "GENERATE SEGMENT";
                }, 1500);
            }
        });
    }, 1000);
});