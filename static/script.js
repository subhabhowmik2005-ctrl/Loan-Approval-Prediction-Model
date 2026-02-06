document.getElementById("loan-form").addEventListener("submit", async function (e) {
    e.preventDefault(); // Stop page reload

    const formData = new FormData(this);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        console.log("Server Response:", result);

        if (result.error) {
            alert(result.error);
            return;
        }

        // 🔹 Show result box
        const resultBox = document.getElementById("result-box");
        resultBox.classList.remove("hidden");

        // 🔹 Update accuracy
        document.getElementById("accuracy-val").innerText = "97%";

        // 🔹 Update status text
        const statusText = document.getElementById("status-text");
        const statusIcon = document.getElementById("status-icon");

        statusText.innerText = result.status;

        // 🔹 Change style based on approval
        if (result.status === "Approved") {
            statusText.style.color = "#00c853";
            statusIcon.innerHTML = "✔️";
        } else {
            statusText.style.color = "#ff1744";
            statusIcon.innerHTML = "❌";
        }

    } catch (error) {
        console.error("Error:", error);
        alert("Something went wrong connecting to the server.");
    }
});


function confettiEffect() {
    // Simple confetti animation could be added here
    // For now, we rely on the CSS animations
}
