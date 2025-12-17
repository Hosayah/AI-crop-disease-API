function uploadImage() {
    const input = document.getElementById("imageInput");
    const result = document.getElementById("result");

    if (!input.files.length) {
        alert("Please select an image");
        return;
    }

    const formData = new FormData();
    formData.append("image", input.files[0]);

    result.innerHTML = "<p>⏳ Analyzing image...</p>";

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Primary result
        let html = `
            <h2>Result</h2>
            <p><strong>Plant:</strong> ${data.plant}</p>
            <p><strong>Disease:</strong> ${data.disease}</p>
            <p><strong>Confidence:</strong> ${data.confidence}%</p>
        `;

        // Warn user if confidence is low
        if (data.confidence < 60) {
            html += `
                <p style="color: orange;">
                    ⚠️ Low confidence prediction. Result may be inaccurate.
                </p>
            `;
        }

        // Top-3 predictions
        if (data.top_predictions && data.top_predictions.length) {
            html += `<h3>Top Predictions</h3><ul>`;
            data.top_predictions.forEach(pred => {
                html += `
                    <li>
                        ${pred.plant} – ${pred.disease}
                        (${pred.confidence}%)
                    </li>
                `;
            });
            html += `</ul>`;
        }

        result.innerHTML = html;
    })
    .catch(err => {
        result.innerHTML = "<p style='color:red;'>❌ Error occurred</p>";
        console.error(err);
    });
}
