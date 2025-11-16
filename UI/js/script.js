const API_URL = "http://127.0.0.1:8000/predict"; // FastAPI endpoint

function sendImage() {
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];
    if (!file) return alert("Please choose an image!");

    // preview image
    const reader = new FileReader();
    reader.onload = function (e) {
        document.getElementById("preview").innerHTML =
            `<img src="${e.target.result}" />`;
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append("file", file);

    document.getElementById("result").innerHTML = "Predicting...";

    fetch(API_URL, {
        method: "POST",
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                document.getElementById("result").innerHTML = "Error: " + data.error;
            } else {
                document.getElementById("result").innerHTML = `
                    <h3>Result:</h3>
                    <p><strong>Breed:</strong> ${data.label}</p>
                    <p><strong>Confidence:</strong> ${data.confidence}</p>
                `;
            }
        })
        .catch(err => {
            console.error(err);
            document.getElementById("result").innerHTML = "Request failed!";
        });
}
