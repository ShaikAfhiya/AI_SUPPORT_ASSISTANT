
async function submitQuery() {

    const query = document.getElementById("customerQuery").value.trim();

    const button = document.getElementById("submitBtn");
    const loading = document.getElementById("loading");
    const resultSection = document.getElementById("resultSection");

    if (!query) {
        alert("Please enter a customer query.");
        return;
    }

    button.disabled = true;
    loading.classList.remove("hidden");
    resultSection.classList.add("hidden");

    try {

        // Send customer query to FastAPI
        const response = await fetch("/support", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                query: query
            })

        });

        const data = await response.json();

        // Display response
        document.getElementById("status").textContent =
            data.status || "ERROR";

        document.getElementById("reply").textContent =
            data.reply || data.detail?.message || "Unable to generate reply.";

        document.getElementById("issue").textContent =
            "Issue: " + (data.issue || "Not identified");

        document.getElementById("orderId").textContent =
            "Order ID: " + (data.order_id || "Not available");

        document.getElementById("evidence").textContent =
            JSON.stringify(data.evidence || [], null, 2);

        resultSection.classList.remove("hidden");

    } catch (error) {

        alert("Unable to connect to the backend. Please check FastAPI and Ollama.");

        console.error(error);

    } finally {

        button.disabled = false;
        loading.classList.add("hidden");

    }
}
