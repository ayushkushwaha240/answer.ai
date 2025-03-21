document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.getElementById('sendButton');
    const queryInput = document.getElementById('queryInput');
    const responseDiv = document.getElementById('response');

    sendButton.addEventListener('click', () => {
        const searchWord = queryInput.value.trim();

        if (searchWord) {
            sendButton.disabled = true; // Disable the button during the request
            responseDiv.innerText = "Processing..."; // Optional: Indicate processing state

            chrome.runtime.sendMessage(
                { 
                    action: "fetch_rag_output", 
                    data: { query: searchWord } 
                }, 
                response => {
                    if (chrome.runtime.lastError) {
                        responseDiv.innerText = `Error: ${chrome.runtime.lastError.message}`;
                    } else if (response && response.success) {
                        responseDiv.innerText = JSON.stringify(response.data, null, 2);
                    } else {
                        responseDiv.innerText = `Error: ${response.error || 'Unknown error'}`;
                    }

                    sendButton.disabled = false; // Re-enable the button after the request
                }
            );
        } else {
            alert("Please enter a word to search for.");
        }
    });
});
