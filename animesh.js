document.getElementById("image-upload").addEventListener("change", function(event) {
    let file = event.target.files[0];
    let previewContainer = document.getElementById("image-preview");
    let previewImage = document.getElementById("preview-img");
    let analyzeButton = document.getElementById("analyze-button");

    if (file) {
        let reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.style.display = "block"; // Show preview
        };
        reader.readAsDataURL(file);

        analyzeButton.disabled = false; // Enable analyze button
        analyzeButton.classList.add("pulse"); // Add animation
    }
});

document.getElementById("analyze-button").addEventListener("click", function() {
    let progressBar = document.getElementById("progress-bar");
    let value = 0;
    let interval = setInterval(() => {
        if (value >= 100) {
            clearInterval(interval);
            document.querySelector(".output-section").style.display = "block";
        }
        progressBar.value = value;
        value += 5;
    }, 200);
});
