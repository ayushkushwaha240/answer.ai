chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "fetch_rag_output") {
        (async () => {
            try {
                console.log("Data: " + message.data.query);
                const response = await fetch("https://ayushkush2402-answer-ai.hf.space/ask/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ question: message.data.query })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                console.log(result); // Logging the result to the console for debugging
                sendResponse({ success: true, data: result });
            } catch (error) {
                console.error('Error:', error);
                sendResponse({ success: false, error: error.message });
            }
        })(); // Immediately invoke the async function

        // Return true to keep the message channel open for the asynchronous response
        return true;
    }
});
