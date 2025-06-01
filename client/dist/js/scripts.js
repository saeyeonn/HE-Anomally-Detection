document.getElementById("csvUploadForm").addEventListener("submit", function (e) {
    e.preventDefault();
  
    const form = e.target;
    const formData = new FormData(form);
    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "Uploading and processing...";
  
    fetch(form.action, {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        resultDiv.innerHTML = `
          <h4>Prediction Summary</h4>
          <p><strong>Stable:</strong> ${data.prediction_summary.stable}</p>
          <p><strong>Unstable:</strong> ${data.prediction_summary.unstable}</p>
          <h5>Sensor Means:</h5>
          <pre>${JSON.stringify(data.mean_values, null, 2)}</pre>
        `;
      })
      .catch((err) => {
        resultDiv.innerHTML = "<p class='text-danger'>Error processing file.</p>";
        console.error(err);
      });
  });
  